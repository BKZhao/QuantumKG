"""
PrimeKG Subgraph Extraction for MIMIC Tasks

For each task row (diagnoses_icd / prescriptions_atc):
  1. Load patient's event_concepts (UMLS text entities up to context_end)
  2. Map texts → PrimeKG entity indices via pre-built mapping
  3. Extract N-hop subgraph from PrimeKG around those entities
  4. Save per-task subgraph as JSON

Usage:
  python extract_subgraphs.py --task diagnoses_icd --hops 2 --max_tasks 1000
  python extract_subgraphs.py --task prescriptions_atc --hops 2 --workers 8
"""

import os
import json
import pickle
import argparse
import pandas as pd
import networkx as nx
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TASK_INDEX_DIR  = "/data/ehr/MIMIC/data_processed/task_index"
CONCEPTS_DIR    = "/data/ehr/MIMIC/data_processed/concept_sequences/event_concepts"
PRIMEKG_GPICKLE = "/data/literature_and_kg/primeKg/primekg_graph.gpickle"
PRIMEKG_CSV     = "/data/literature_and_kg/primeKg/PrimeKg.csv"

OUTPUT_BASE  = "/data/MIMIC/primekg_subgraphs"
MAPPING_FILE = os.path.join(OUTPUT_BASE, "umls_to_primekg.pkl")
NODE_ATTRS_FILE = os.path.join(OUTPUT_BASE, "primekg_node_attrs.pkl")

# 只保留这些有意义的关系类型（排除超密集的protein_protein 等）
ALLOWED_RELATIONS = {
    "indication",                      # 药物→疾病适应症
    "contraindication",                # 药物→禁忌症
    "off-label use",                   # 药物→非标签使用
    "disease_phenotype_positive",      # 疾病→表型
    "disease_phenotype_negative",      # 疾病→表型（负）
    "disease_protein",                 # 疾病→蛋白质
    "drug_protein",                    # 药物→蛋白质
    "drug_effect",                     # 药物→效果
    "disease_disease",                 # 疾病→疾病
    "phenotype_phenotype",             # 表型→表型
    "exposure_disease",                # 暴露→疾病
    "drug_drug",                       # 药物→药物协同作用
}

DISEASE_FIELDS = {"diagnoses_icd", "diagnoses_ccs", "diagnosis", "diagnosis_ccs"}
DRUG_FIELDS    = {"prescriptions", "emar", "medrecon"}
RELEVANT_FIELDS = DISEASE_FIELDS | DRUG_FIELDS


# ---------------------------------------------------------------------------
# Load PrimeKG graph
# ---------------------------------------------------------------------------

def load_primekg_graph():
    """Load PrimeKG as a NetworkX graph. Prefer gpickle; fall back to CSV."""
    if os.path.exists(PRIMEKG_GPICKLE):
        print(f"Loading PrimeKG from gpickle ({PRIMEKG_GPICKLE})...")
        with open(PRIMEKG_GPICKLE, "rb") as f:
            G = pickle.load(f)
        print(f"  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
        return G

    print(f"Loading PrimeKG from CSV ({PRIMEKG_CSV})...")
    df = pd.read_csv(PRIMEKG_CSV, low_memory=False)
    G = nx.Graph()
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  building graph"):
        u = int(row["x_index"])
        v = int(row["y_index"])
        G.add_node(u, name=row["x_name"], node_type=row["x_type"],
                   source=row["x_source"], node_id=row["x_id"])
        G.add_node(v, name=row["y_name"], node_type=row["y_type"],
                   source=row["y_source"], node_id=row["y_id"])
        G.add_edge(u, v, relation=row["relation"],
                   display_relation=row["display_relation"])
    print(f"  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


# ---------------------------------------------------------------------------
# Build PrimeKG name→node_index lookup
# ---------------------------------------------------------------------------

def build_primekg_name_index(G: nx.Graph) -> dict:
    """name.lower() → list of node indices"""
    idx = {}
    for node, data in G.nodes(data=True):
        name = data.get("name", "")
        if name:
            idx.setdefault(name.lower(), []).append(node)
    return idx


# ---------------------------------------------------------------------------
# Subgraph extraction
# ---------------------------------------------------------------------------

def extract_n_hop_subgraph(G: nx.Graph, seed_nodes: list, hops: int, node_attrs: dict = None) -> dict:
    """
    Extract the N-hop ego subgraph around seed_nodes.
    Only includes edges with allowed relation types.
    Returns a dict with 'nodes' and 'edges' lists.

    Args:
        G: PrimeKG graph
        seed_nodes: list of seed node indices
        hops: number of hops
        node_attrs: dict mapping node_index (str) -> {name, type, source, id} for enriching nodes
    """
    if not seed_nodes:
        return {"nodes": [], "edges": []}

    # Convert seed_nodes to strings (graph nodes are strings)
    seed_nodes_str = {str(n) for n in seed_nodes}

    # BFS to collect all nodes within `hops` steps (only via allowed relations)
    frontier = seed_nodes_str & set(G.nodes)
    visited = set(frontier)

    for _ in range(hops):
        nxt = set()
        for n in frontier:
            for neighbor in G.neighbors(n):
                # Check if any edge between n and neighbor has allowed relation
                edge_data = G.get_edge_data(n, neighbor)
                if edge_data and edge_data.get('relation') in ALLOWED_RELATIONS:
                    nxt.add(neighbor)
        nxt -= visited
        visited.update(nxt)
        frontier = nxt

    sub = G.subgraph(visited)

    # Build nodes with attributes from node_attrs dict
    nodes = []
    for n, data in sub.nodes(data=True):
        node_dict = {"index": n}
        # Add attributes from graph (usually empty for gpickle)
        node_dict.update({k: v for k, v in data.items()})
        # Enrich with attributes from CSV if available
        if node_attrs and n in node_attrs:
            node_dict.update(node_attrs[n])
        nodes.append(node_dict)

    # Only include edges with allowed relations AND at least one endpoint is a seed node
    # This prevents including all edges between 1-hop neighbors (which causes combinatorial explosion)
    edges = [
        {"source": u, "target": v, **{k: v2 for k, v2 in data.items()}}
        for u, v, data in sub.edges(data=True)
        if data.get('relation') in ALLOWED_RELATIONS and (u in seed_nodes_str or v in seed_nodes_str)
    ]

    # Add drug_drug edges between seed nodes (synergistic interactions)
    # These are not used for BFS expansion but provide valuable drug interaction info
    seed_nodes_in_graph = seed_nodes_str & set(G.nodes)
    seed_list = sorted(seed_nodes_in_graph)
    for i, seed1 in enumerate(seed_list):
        for seed2 in seed_list[i+1:]:
            if G.has_edge(seed1, seed2):
                edge_data = G.get_edge_data(seed1, seed2)
                if edge_data and edge_data.get('relation') == 'drug_drug':
                    # Add the edge regardless of whether seeds are in visited
                    # (they are seed nodes, so they should be included)
                    edges.append({"source": seed1, "target": seed2, **{k: v for k, v in edge_data.items()}})
                    # Ensure both seed nodes are in the node list
                    if seed1 not in visited:
                        visited.add(seed1)
                        node_dict = {"index": seed1}
                        if node_attrs and seed1 in node_attrs:
                            node_dict.update(node_attrs[seed1])
                        nodes.append(node_dict)
                    if seed2 not in visited:
                        visited.add(seed2)
                        node_dict = {"index": seed2}
                        if node_attrs and seed2 in node_attrs:
                            node_dict.update(node_attrs[seed2])
                        nodes.append(node_dict)

    return {"nodes": nodes, "edges": edges, "seed_nodes": list(seed_nodes_str)}



# ---------------------------------------------------------------------------
# Build and cache node attributes
# ---------------------------------------------------------------------------

def build_and_cache_node_attrs():
    """Build node attributes dict from CSV and cache it."""
    if os.path.exists(NODE_ATTRS_FILE):
        print(f"Loading cached node attributes from {NODE_ATTRS_FILE}...")
        with open(NODE_ATTRS_FILE, "rb") as f:
            node_attrs = pickle.load(f)
        print(f"  {len(node_attrs):,} nodes with attributes")
        return node_attrs

    print(f"Building node attributes from CSV ({PRIMEKG_CSV})...")
    node_attrs = {}  # node_index (str) -> {name, type, source, id}
    df_kg = pd.read_csv(PRIMEKG_CSV, low_memory=False)
    for side in [("x_index", "x_name", "x_type", "x_source", "x_id"),
                 ("y_index", "y_name", "y_type", "y_source", "y_id")]:
        idx_col, name_col, type_col, src_col, id_col = side
        for _, row in df_kg[[idx_col, name_col, type_col, src_col, id_col]].drop_duplicates(subset=[idx_col]).iterrows():
            node_idx = str(int(row[idx_col]))
            if node_idx not in node_attrs:
                node_attrs[node_idx] = {
                    "name": row[name_col],
                    "node_type": row[type_col],
                    "source": row[src_col],
                    "node_id": row[id_col]
                }
    print(f"  {len(node_attrs):,} nodes with attributes")

    # Cache it
    print(f"Caching node attributes to {NODE_ATTRS_FILE}...")
    with open(NODE_ATTRS_FILE, "wb") as f:
        pickle.dump(node_attrs, f)

    return node_attrs


# ---------------------------------------------------------------------------
# Patient concept loading
# ---------------------------------------------------------------------------

def load_patient_concepts(subject_id: int, context_end: int = None) -> list:
    """
    Load UMLS text entities for a patient (only disease and drug fields).
    If context_end is specified, only load the first context_end events.
    """
    fpath = os.path.join(CONCEPTS_DIR, f"{subject_id}.pkl")
    if not os.path.exists(fpath):
        return []
    try:
        with open(fpath, "rb") as f:
            events = pickle.load(f)

        # Only use first context_end events if specified
        if context_end is not None and isinstance(events, list):
            events = events[:context_end]

        texts = []
        for event in events:
            concepts = event.get("concepts", {})
            if not isinstance(concepts, dict):
                continue
            for field, terms in concepts.items():
                if field not in RELEVANT_FIELDS:
                    continue
                if not isinstance(terms, list):
                    continue
                texts.extend(t.strip() for t in terms if t and isinstance(t, str))
        return texts
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Worker function for multiprocessing
# ---------------------------------------------------------------------------

# Global variables for worker processes
_G = None
_umls_mapping = None
_task_name = None
_hops = None
_out_dir = None

def init_worker(task_name, hops, out_dir):
    """Initialize worker process with shared data."""
    global _G, _umls_mapping, _task_name, _hops, _out_dir

    _task_name = task_name
    _hops = hops
    _out_dir = out_dir

    # Load mapping
    with open(MAPPING_FILE, "rb") as f:
        _umls_mapping = pickle.load(f)

    # Load graph
    _G = load_primekg_graph()


def process_single_task(row_data):
    """Process a single task (called by worker process)."""
    subject_id, hadm_id, context_end, target = row_data
    task_id = f"{subject_id}_{hadm_id}"

    # Get patient's UMLS text entities (only up to context_end)
    texts = load_patient_concepts(subject_id, context_end)

    # Map to PrimeKG node indices
    seed_nodes = set()
    mapped_entities = []
    for text in texts:
        entry = _umls_mapping.get(text)
        if not entry or not entry.get("primekg"):
            continue
        pkg_entity = entry["primekg"]
        node_idx = pkg_entity.get("index")
        if node_idx is None:
            continue
        seed_nodes.add(node_idx)
        mapped_entities.append({
            "text": text,
            "cui": entry.get("cui"),
            "primekg_name": pkg_entity.get("name"),
            "primekg_type": pkg_entity.get("type"),
            "primekg_source": pkg_entity.get("source"),
            "par_depth": entry.get("par_depth", -1),
            "node_index": node_idx,
        })

    if not seed_nodes:
        has_subgraph = False
        nodes = []
        edges = []
    else:
        has_subgraph = True

        # Collect all nodes (seed + intermediate nodes from paths)
        all_nodes = set(str(n) for n in seed_nodes)

        # Collect all edges (direct edges + path edges)
        edges = []

        # Track which seed pairs already have direct edges
        direct_pairs = set()

        seed_list = sorted(seed_nodes)

        # Step 1: Add direct edges between seed nodes
        for i, n1 in enumerate(seed_list):
            for n2 in seed_list[i+1:]:
                n1_str, n2_str = str(n1), str(n2)
                if _G.has_edge(n1_str, n2_str):
                    edge_data = _G.get_edge_data(n1_str, n2_str)
                    if edge_data and edge_data.get('relation') in ALLOWED_RELATIONS:
                        edges.append([n1_str, n2_str, edge_data['relation']])
                        direct_pairs.add((min(n1_str, n2_str), max(n1_str, n2_str)))

        # Step 2: Add shortest paths (≤2-hop) for seed pairs without direct edges
        for i, n1 in enumerate(seed_list):
            for n2 in seed_list[i+1:]:
                n1_str, n2_str = str(n1), str(n2)

                # Skip if already has direct edge
                pair_key = (min(n1_str, n2_str), max(n1_str, n2_str))
                if pair_key in direct_pairs:
                    continue

                # Try to find shortest path (max 2 hops)
                try:
                    if n1_str in _G and n2_str in _G:
                        path = nx.shortest_path(_G, n1_str, n2_str)
                        path_length = len(path) - 1  # number of edges

                        if path_length <= 3:
                            # Add intermediate nodes
                            for node in path:
                                all_nodes.add(node)

                            # Add edges along the path
                            for j in range(len(path) - 1):
                                src, dst = path[j], path[j+1]
                                edge_data = _G.get_edge_data(src, dst)
                                if edge_data:
                                    edge = [src, dst, edge_data.get('relation', 'unknown')]
                                    # Avoid duplicate edges
                                    if edge not in edges:
                                        edges.append(edge)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    # No path or node not in graph, skip
                    pass

        nodes = list(all_nodes)

    # Minimal output format - only store indices
    output = {
        "task_id": task_id,
        "subject_id": subject_id,
        "hadm_id": hadm_id,
        "context_end": context_end,
        "task": _task_name,
        "target": target,
        "seed_nodes": [int(n) for n in seed_nodes] if seed_nodes else [],  # Original seed nodes
        "nodes": [int(n) if isinstance(n, int) else int(n) for n in nodes],  # All nodes (seed + intermediate)
        "edges": edges,
    }

    # Use context_end in filename to ensure uniqueness (same patient/hadm can have multiple time points)
    filename = f"{task_id}_{context_end}.json" if context_end is not None else f"{task_id}.json"
    out_file = os.path.join(_out_dir, filename)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)

    return has_subgraph


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_task(task_name: str, hops: int, max_tasks: int | None, workers: int = 1):
    task_csv = os.path.join(TASK_INDEX_DIR, f"{task_name}.csv")
    if not os.path.exists(task_csv):
        raise FileNotFoundError(f"Task index not found: {task_csv}")

    # Load task index
    print(f"Loading task index: {task_csv}")
    df = pd.read_csv(task_csv, on_bad_lines="skip")
    if max_tasks:
        df = df.head(max_tasks)
    print(f"  {len(df):,} tasks to process")
    print(f"  Using {workers} worker(s)")

    # Output directory
    out_dir = os.path.join(OUTPUT_BASE, task_name)
    os.makedirs(out_dir, exist_ok=True)

    # Prepare task data
    task_data = []
    for _, row in df.iterrows():
        subject_id = int(row["subject_id"])
        hadm_id = str(row.get("hadm_id", ""))
        context_end = int(row.get("context_end", 0)) if pd.notna(row.get("context_end")) else None
        target = row.get("target")
        task_data.append((subject_id, hadm_id, context_end, target))

    # Process with multiprocessing
    stats = {"total": len(task_data), "with_subgraph": 0, "empty": 0}

    if workers == 1:
        # Single process mode (for debugging)
        init_worker(task_name, hops, out_dir)
        for data in tqdm(task_data, desc=f"  {task_name}"):
            has_subgraph = process_single_task(data)
            if has_subgraph:
                stats["with_subgraph"] += 1
            else:
                stats["empty"] += 1
    else:
        # Multi-process mode
        with Pool(processes=workers, initializer=init_worker, initargs=(task_name, hops, out_dir)) as pool:
            for has_subgraph in tqdm(pool.imap_unordered(process_single_task, task_data),
                                     total=len(task_data), desc=f"  {task_name}"):
                if has_subgraph:
                    stats["with_subgraph"] += 1
                else:
                    stats["empty"] += 1

    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"  Total processed : {stats['total']:,}")
    print(f"  With subgraph   : {stats['with_subgraph']:,}")
    print(f"  Empty (no match): {stats['empty']:,}")
    print(f"  Output dir      : {out_dir}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract PrimeKG subgraphs for MIMIC tasks")
    parser.add_argument("--task", required=True,
                        choices=["diagnoses_icd", "prescriptions_atc"],
                        help="Task name")
    parser.add_argument("--hops", type=int, default=2,
                        help="N-hop neighborhood size (default: 2)")
    parser.add_argument("--max_tasks", type=int, default=None,
                        help="Limit number of tasks (for testing)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of worker processes (default: 1)")
    args = parser.parse_args()

    process_task(args.task, args.hops, args.max_tasks, args.workers)


if __name__ == "__main__":
    main()
