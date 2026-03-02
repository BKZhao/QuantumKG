"""
UMLS 文本 → PrimeKG 实体映射（全内存版，速度快）

策略：
  - 启动时一次性把 UMLS strings / PAR relations 全量加载到内存
  - 映射时完全走字典查找，不走 MongoDB 网络查询
  - PrimeKG 实体从 CSV 读取（节点属性完整）

只处理两类字段：
  - 诊断字段 (diagnoses_icd / diagnoses_ccs / diagnosis / diagnosis_ccs)
      → 文本 → UMLS CUI → MONDO/HPO → PrimeKG disease/phenotype 节点
      → 找不到时沿 PAR 层级向上找父疾病节点（最多 MAX_PAR_DEPTH 层）
  - 药物字段 (prescriptions / emar / medrecon)
      → 文本 → UMLS CUI → DrugBank → PrimeKG drug 节点
      → 找不到时直接 skip

输出: outputs/umls_to_primekg.pkl
  {
    text (str) → {
        'cui':       str | None,
        'primekg':   {id, name, type, source, ...} | None,
        'par_depth': int,   # 0=直接匹配, >0=通过祖先, -1=未找到
        'field_type': 'disease' | 'drug'
    }
  }
"""

import os
import pickle
import glob
from collections import defaultdict

import pandas as pd
from pymongo import MongoClient
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.environ.get("MONGO_DB_NAME", "umls_test")

PRIMEKG_CSV        = "/data/literature_and_kg/primeKg/PrimeKg.csv"
EVENT_CONCEPTS_DIR = "/data/ehr/MIMIC/data_processed/concept_sequences/event_concepts"
OUTPUT_DIR  = "/data/MIMIC/primekg_subgraphs"
MAPPING_FILE = os.path.join(OUTPUT_DIR, "umls_to_primekg.pkl")

MAX_PAR_DEPTH = 5

DISEASE_FIELDS = {"diagnoses_icd", "diagnoses_ccs", "diagnosis", "diagnosis_ccs"}
DRUG_FIELDS    = {"prescriptions", "emar", "medrecon"}


# ---------------------------------------------------------------------------
# ID 格式转换
# ---------------------------------------------------------------------------

def umls_to_primekg_id(sab: str, code: str):
    s = sab.upper()
    if s == "DRUGBANK":
        return (code, "DrugBank")
    if s == "HPO":
        num = code.replace("HP:", "").lstrip("0") or "0"
        return (num, "HPO")
    return None


# ---------------------------------------------------------------------------
# [1/4] 从 CSV 加载 PrimeKG 实体索引
# ---------------------------------------------------------------------------

def load_primekg_index_from_csv(csv_path: str):
    print("\n[1/4] 从 CSV 加载 PrimeKG 实体索引...")
    id_source_to_entity = {}   # (id, source) → entity dict
    name_to_entities    = defaultdict(list)  # name.lower() → [entity dicts]

    df = pd.read_csv(csv_path, low_memory=False,
                     usecols=["x_index","x_id","x_type","x_name","x_source",
                               "y_index","y_id","y_type","y_name","y_source"])

    seen = set()
    for side in [("x_index","x_id","x_type","x_name","x_source"),
                 ("y_index","y_id","y_type","y_name","y_source")]:
        idx_col, id_col, type_col, name_col, src_col = side
        sub = df[[idx_col, id_col, type_col, name_col, src_col]].drop_duplicates(subset=[idx_col])
        for _, row in tqdm(sub.iterrows(), total=len(sub), desc=f"  {side[0]}"):
            node_idx = int(row[idx_col])
            if node_idx in seen:
                continue
            seen.add(node_idx)
            entity = {
                "index":  node_idx,
                "id":     str(row[id_col]),
                "type":   row[type_col],
                "name":   row[name_col],
                "source": row[src_col],
            }
            key = (str(row[id_col]), row[src_col])
            id_source_to_entity[key] = entity
            if row[name_col]:
                name_to_entities[str(row[name_col]).lower()].append(entity)

    print(f"  {len(id_source_to_entity):,} 个实体，{len(name_to_entities):,} 个名字索引")
    return id_source_to_entity, name_to_entities


# ---------------------------------------------------------------------------
# [2/4] 全量加载 UMLS strings 到内存
# ---------------------------------------------------------------------------

def load_umls_to_memory(db):
    """
    返回三个字典：
      name_lower_to_cuis : source_name.lower() → [(cui, is_preferred), ...]
      cui_to_records     : cui → [(source, source_code, is_preferred), ...]
      cui_to_pref_name   : cui → preferred source_name（is_preferred=Y 的第一个）
    """
    print("\n[2/4] 全量加载 UMLS strings 到内存（8.6M 条，约需 2-3 分钟）...")
    col = db["umls_strings_raw_test"]

    name_lower_to_cuis = defaultdict(list)   # name.lower() → [(cui, is_pref)]
    cui_to_records     = defaultdict(list)   # cui → [(source, source_code, is_pref)]
    cui_to_pref_name   = {}                  # cui → preferred name

    for doc in tqdm(col.find({}, {"cui":1,"source_name":1,"source":1,"source_code":1,"is_preferred":1,"_id":0}),
                    total=col.estimated_document_count(), desc="  umls_strings"):
        cui      = doc.get("cui", "")
        name     = doc.get("source_name", "")
        source   = doc.get("source", "")
        code     = doc.get("source_code", "")
        is_pref  = doc.get("is_preferred", "") == "Y"

        # 过滤掉非字符串字段（MongoDB 里有些是 float/NaN）
        if name and isinstance(name, str):
            name_lower_to_cuis[name.lower()].append((cui, is_pref))
        if cui and isinstance(cui, str):
            cui_to_records[cui].append((source, code, is_pref))
            if is_pref and cui not in cui_to_pref_name and isinstance(name, str):
                cui_to_pref_name[cui] = name

    print(f"  {len(name_lower_to_cuis):,} 个唯一名字，{len(cui_to_records):,} 个唯一 CUI")
    return name_lower_to_cuis, cui_to_records, cui_to_pref_name


# ---------------------------------------------------------------------------
# [3/4] 全量加载 PAR 关系到内存
# ---------------------------------------------------------------------------

def load_par_relations(db):
    """cui → set of parent cuis"""
    print("\n[3/4] 加载 PAR 关系到内存（27M 条，只取 PAR，约需 2-3 分钟）...")
    col = db["umls_relations_all"]
    par = defaultdict(set)
    total = col.count_documents({"rel": "PAR"})
    for doc in tqdm(col.find({"rel": "PAR"}, {"cui1":1,"cui2":1,"_id":0}),
                    total=total, desc="  PAR relations"):
        par[doc["cui1"]].add(doc["cui2"])
    print(f"  {len(par):,} 个 CUI 有 PAR 父节点")
    return par


# ---------------------------------------------------------------------------
# 收集唯一文本
# ---------------------------------------------------------------------------

def collect_unique_texts(concepts_dir: str):
    print("\n收集诊断和药物文本...")
    disease_texts = set()
    drug_texts    = set()

    pkl_files = glob.glob(os.path.join(concepts_dir, "*.pkl"))
    for fpath in tqdm(pkl_files, desc="  pkl 文件"):
        try:
            with open(fpath, "rb") as f:
                events = pickle.load(f)
            if not isinstance(events, list):
                continue
            for ev in events:
                if not isinstance(ev, dict):
                    continue
                concepts = ev.get("concepts", {})
                if not isinstance(concepts, dict):
                    continue
                for field, terms in concepts.items():
                    if not isinstance(terms, list):
                        continue
                    for t in terms:
                        if not isinstance(t, str) or not t.strip():
                            continue
                        t = t.strip()
                        if field in DISEASE_FIELDS:
                            disease_texts.add(t)
                        elif field in DRUG_FIELDS:
                            drug_texts.add(t)
        except Exception:
            pass

    print(f"  诊断文本: {len(disease_texts):,} 个唯一词条")
    print(f"  药物文本: {len(drug_texts):,} 个唯一词条")
    return disease_texts, drug_texts


# ---------------------------------------------------------------------------
# 核心映射函数（纯内存）
# ---------------------------------------------------------------------------

def text_to_cui(text: str, name_lower_to_cuis: dict) -> str | None:
    """文本 → UMLS CUI，优先 is_preferred=Y"""
    hits = name_lower_to_cuis.get(text.lower())
    if not hits:
        return None
    for cui, is_pref in hits:
        if is_pref:
            return cui
    return hits[0][0]


def cui_to_primekg_drug(cui: str, cui_to_records: dict, id_source_to_entity: dict):
    for source, code, _ in cui_to_records.get(cui, []):
        if source.upper() == "DRUGBANK":
            key = umls_to_primekg_id(source, code)
            if key and key in id_source_to_entity:
                return id_source_to_entity[key]
    return None


def cui_to_primekg_disease(cui: str, cui_to_records: dict,
                            cui_to_pref_name: dict,
                            id_source_to_entity: dict,
                            name_to_entities: dict):
    # HPO 精确 ID 匹配
    for source, code, _ in cui_to_records.get(cui, []):
        if source.upper() == "HPO":
            key = umls_to_primekg_id(source, code)
            if key and key in id_source_to_entity:
                return id_source_to_entity[key]

    # preferred name → PrimeKG name 匹配（覆盖 MONDO）
    pref_name = cui_to_pref_name.get(cui)
    if pref_name:
        hits = name_to_entities.get(pref_name.lower())
        if hits:
            for h in hits:
                if h.get("type") == "disease":
                    return h
            return hits[0]

    return None


def map_disease(text, name_lower_to_cuis, cui_to_records, cui_to_pref_name, par_map,
                id_source_to_entity, name_to_entities):
    result = {"cui": None, "primekg": None, "par_depth": -1, "field_type": "disease"}

    cui = text_to_cui(text, name_lower_to_cuis)
    if not cui:
        return result
    result["cui"] = cui

    entity = cui_to_primekg_disease(cui, cui_to_records, cui_to_pref_name,
                                     id_source_to_entity, name_to_entities)
    if entity:
        result["primekg"] = entity
        result["par_depth"] = 0
        return result

    # PAR 向上回退
    current = {cui}
    visited = {cui}
    for depth in range(1, MAX_PAR_DEPTH + 1):
        nxt = set()
        for c in current:
            nxt.update(par_map.get(c, set()) - visited)
        visited.update(nxt)
        if not nxt:
            break
        for p in nxt:
            entity = cui_to_primekg_disease(p, cui_to_records, cui_to_pref_name,
                                             id_source_to_entity, name_to_entities)
            if entity:
                result["primekg"] = entity
                result["par_depth"] = depth
                return result
        current = nxt

    return result


def map_drug(text, name_lower_to_cuis, cui_to_records, id_source_to_entity):
    result = {"cui": None, "primekg": None, "par_depth": -1, "field_type": "drug"}

    cui = text_to_cui(text, name_lower_to_cuis)
    if not cui:
        return result
    result["cui"] = cui

    entity = cui_to_primekg_drug(cui, cui_to_records, id_source_to_entity)
    if entity:
        result["primekg"] = entity
        result["par_depth"] = 0

    return result


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载 PrimeKG（从 CSV）
    id_source_to_entity, name_to_entities = load_primekg_index_from_csv(PRIMEKG_CSV)

    # 全量加载 UMLS 到内存
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10000)
    db = client[DB_NAME]
    name_lower_to_cuis, cui_to_records, cui_to_pref_name = load_umls_to_memory(db)
    par_map = load_par_relations(db)
    client.close()

    # 收集文本
    disease_texts, drug_texts = collect_unique_texts(EVENT_CONCEPTS_DIR)

    print(f"\n[4/4] 开始映射（纯内存，无网络查询）...")
    mapping = {}

    print(f"  映射 {len(disease_texts):,} 个诊断词条（含 PAR 回退）...")
    for text in tqdm(sorted(disease_texts), desc="  诊断"):
        mapping[text] = map_disease(text, name_lower_to_cuis, cui_to_records, cui_to_pref_name,
                                    par_map, id_source_to_entity, name_to_entities)

    print(f"  映射 {len(drug_texts):,} 个药物词条...")
    for text in tqdm(sorted(drug_texts), desc="  药物"):
        mapping[text] = map_drug(text, name_lower_to_cuis, cui_to_records, id_source_to_entity)

    with open(MAPPING_FILE, "wb") as f:
        pickle.dump(mapping, f)
    print(f"\n映射表已保存 → {MAPPING_FILE}")

    # 统计
    d_total  = sum(1 for v in mapping.values() if v["field_type"] == "disease")
    d_found  = sum(1 for v in mapping.values() if v["field_type"] == "disease" and v["primekg"])
    d_direct = sum(1 for v in mapping.values() if v["field_type"] == "disease" and v["par_depth"] == 0)
    d_par    = sum(1 for v in mapping.values() if v["field_type"] == "disease" and v["par_depth"] > 0)
    dr_total = sum(1 for v in mapping.values() if v["field_type"] == "drug")
    dr_found = sum(1 for v in mapping.values() if v["field_type"] == "drug" and v["primekg"])

    print(f"\n{'='*60}")
    print(f"诊断词条: {d_total:,}")
    print(f"  映射成功: {d_found:,} ({d_found/d_total*100:.1f}%)")
    print(f"    直接匹配 (depth=0): {d_direct:,}")
    print(f"    PAR 祖先匹配:       {d_par:,}")
    print(f"药物词条: {dr_total:,}")
    print(f"  映射成功: {dr_found:,} ({dr_found/dr_total*100:.1f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
