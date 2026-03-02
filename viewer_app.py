#!/usr/bin/env python3
"""
MIMIC PrimeKG子图可视化Web应用
使用Streamlit创建交互式界面
"""
import streamlit as st
import json
import pickle
import os
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import networkx as nx

# 路径配置
SUBGRAPH_BASE = "/data/MIMIC/primekg_subgraphs"
NODE_ATTRS_CACHE = "/data/MIMIC/primekg_subgraphs/primekg_node_attrs.pkl"


# 页面配置
st.set_page_config(
    page_title="MIMIC PrimeKG子图浏览器",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .info-box {
        background: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_node_attributes():
    """加载PrimeKG节点属性"""
    with open(NODE_ATTRS_CACHE, 'rb') as f:
        return pickle.load(f)

@st.cache_data
def get_file_list(task_type):
    """获取指定任务类型的所有文件"""
    task_dir = os.path.join(SUBGRAPH_BASE, task_type)
    files = list(Path(task_dir).glob('*.json'))

    # 提取文件信息
    file_info = []
    for f in files:
        parts = f.stem.split('_')
        if len(parts) >= 3:
            subject_id = parts[0]
            hadm_id = parts[1]
            context_end = parts[2]
            file_info.append({
                'filename': f.name,
                'subject_id': subject_id,
                'hadm_id': hadm_id,
                'context_end': context_end,
                'path': str(f)
            })

    return pd.DataFrame(file_info)

def load_subgraph(json_path, node_attrs):
    """加载子图数据"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    seed_nodes = set(data.get('seed_nodes', []))
    all_nodes = data.get('nodes', data.get('seed_nodes', []))
    edges = data.get('edges', [])

    return data, seed_nodes, all_nodes, edges

def create_path_visualization(path_info, all_nodes, edges, node_attrs, seed_nodes):
    """为特定路径创建可视化"""
    # 创建NetworkX图
    G = nx.DiGraph()

    # 只添加路径中的节点
    path_nodes = path_info['path']
    for node_id in path_nodes:
        node_str = str(node_id)
        attrs = node_attrs.get(node_str, {})
        is_seed = node_id in seed_nodes

        G.add_node(node_id,
                   name=attrs.get('name', 'Unknown'),
                   node_type=attrs.get('node_type', 'Unknown'),
                   source=attrs.get('source', 'Unknown'),
                   is_seed=is_seed)

    # 添加路径中的边
    for i in range(len(path_nodes) - 1):
        src, dst = path_nodes[i], path_nodes[i+1]
        # 从原始edges中找到这条边的关系
        relation = 'connected'
        for edge in edges:
            edge_src = int(edge[0]) if isinstance(edge[0], str) else edge[0]
            edge_dst = int(edge[1]) if isinstance(edge[1], str) else edge[1]
            if edge_src == src and edge_dst == dst:
                relation = edge[2]
                break
        G.add_edge(src, dst, relation=relation)

    # 使用hierarchical layout来显示路径
    pos = {}
    for i, node in enumerate(path_nodes):
        pos[node] = (i * 2, 0)  # 水平排列

    # 创建边的trace
    edge_traces = []
    edge_annotations = []

    for i in range(len(path_nodes) - 1):
        src, dst = path_nodes[i], path_nodes[i+1]
        x0, y0 = pos[src]
        x1, y1 = pos[dst]

        edge_data = G.get_edge_data(src, dst)
        relation = edge_data.get('relation', 'unknown')
        src_name = G.nodes[src]['name']
        dst_name = G.nodes[dst]['name']

        # 边的线条（高亮显示）
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=5, color='#667eea'),
            hoverinfo='text',
            text=f"<b>{relation}</b><br>{src_name} → {dst_name}",
            showlegend=False
        )
        edge_traces.append(edge_trace)

        # 箭头
        edge_annotations.append(
            dict(
                ax=x0, ay=y0,
                x=x1, y=y1,
                xref='x', yref='y',
                axref='x', ayref='y',
                showarrow=True,
                arrowhead=3,
                arrowsize=2.5,
                arrowwidth=3,
                arrowcolor='#667eea',
                standoff=25,
            )
        )

        # 在边的中间添加关系标签
        mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2 + 0.3
        edge_annotations.append(
            dict(
                x=mid_x, y=mid_y,
                text=f"<b>{relation}</b>",
                showarrow=False,
                font=dict(size=12, color='#667eea'),
                bgcolor='rgba(255,255,255,0.8)',
                borderpad=4
            )
        )

    # 创建节点trace
    node_x, node_y, node_text, node_labels, node_colors, node_sizes = [], [], [], [], [], []

    for i, node in enumerate(path_nodes):
        x, y = pos[node]
        attrs = G.nodes[node]
        is_seed = attrs.get('is_seed', False)

        # 节点信息
        text = f"<b>步骤 {i+1}</b><br>"
        text += f"<b>{attrs['name']}</b><br>"
        text += f"Type: {attrs['node_type']}<br>"
        text += f"Source: {attrs['source']}<br>"
        text += f"<b>{'🔴 Seed Node' if is_seed else '🔵 Intermediate Node'}</b>"

        node_x.append(x)
        node_y.append(y)
        node_text.append(text)
        node_labels.append(f"{i+1}. {attrs['name'][:15]}")
        node_colors.append('#FF6B6B' if is_seed else '#4ECDC4')
        node_sizes.append(40 if is_seed else 30)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_labels,
        textposition="top center",
        textfont=dict(size=12, family='Arial'),
        hovertext=node_text,
        marker=dict(
            color=node_colors,
            size=node_sizes,
            line=dict(width=4, color='white'),
            symbol='circle'
        ),
        showlegend=False
    )

    # 创建图形
    fig = go.Figure(data=edge_traces + [node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=40, l=40, r=40, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 1]),
                        height=400,
                        plot_bgcolor='rgba(240,242,246,0.5)',
                        annotations=edge_annotations
                    ))

    return fig

def analyze_paths(seed_nodes, all_nodes, edges):
    """分析seed节点之间的路径"""
    # 构建图
    G = nx.DiGraph()
    for edge in edges:
        src, dst, rel = edge
        src_id = int(src) if isinstance(src, str) else src
        dst_id = int(dst) if isinstance(dst, str) else dst
        G.add_edge(src_id, dst_id, relation=rel)

    # 找出seed节点之间的所有路径
    paths_info = []
    seed_list = list(seed_nodes)

    for i, src in enumerate(seed_list):
        for dst in seed_list[i+1:]:
            try:
                if nx.has_path(G, src, dst):
                    # 找最短路径
                    path = nx.shortest_path(G, src, dst)
                    path_length = len(path) - 1
                    paths_info.append({
                        'source': src,
                        'target': dst,
                        'path': path,
                        'length': path_length
                    })
            except:
                pass

    return paths_info

def create_network_graph(data, seed_nodes, all_nodes, edges, node_attrs):
    """创建网络图可视化"""

    # 创建NetworkX图
    G = nx.DiGraph()

    # 添加节点
    for node_id in all_nodes:
        node_str = str(node_id)
        attrs = node_attrs.get(node_str, {})
        is_seed = node_id in seed_nodes

        G.add_node(node_id,
                   name=attrs.get('name', 'Unknown'),
                   node_type=attrs.get('node_type', 'Unknown'),
                   source=attrs.get('source', 'Unknown'),
                   is_seed=is_seed)

    # 添加边
    for edge in edges:
        src, dst, rel = edge
        src_id = int(src) if isinstance(src, str) else src
        dst_id = int(dst) if isinstance(dst, str) else dst
        G.add_edge(src_id, dst_id, relation=rel)

    # 使用spring layout，增加节点间距
    pos = nx.spring_layout(G, k=4, iterations=150, seed=42)

    # 创建边的trace（带箭头和详细信息）
    edge_traces = []
    edge_annotations = []

    for edge in G.edges(data=True):
        src, dst, edge_data = edge
        x0, y0 = pos[src]
        x1, y1 = pos[dst]

        relation = edge_data.get('relation', 'unknown')
        src_name = G.nodes[src]['name']
        dst_name = G.nodes[dst]['name']
        src_type = G.nodes[src]['node_type']
        dst_type = G.nodes[dst]['node_type']

        # 边的hover信息
        hover_text = f"<b>关系: {relation}</b><br>"
        hover_text += f"<br><b>源节点:</b><br>"
        hover_text += f"  名称: {src_name}<br>"
        hover_text += f"  类型: {src_type}<br>"
        hover_text += f"  ID: {src}<br>"
        hover_text += f"<br><b>目标节点:</b><br>"
        hover_text += f"  名称: {dst_name}<br>"
        hover_text += f"  类型: {dst_type}<br>"
        hover_text += f"  ID: {dst}<br>"

        # 边的线条（加粗，更明显）
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=3, color='rgba(150,150,150,0.6)'),
            hoverinfo='text',
            text=hover_text,
            showlegend=False
        )
        edge_traces.append(edge_trace)

        # 箭头（使用annotation）
        edge_annotations.append(
            dict(
                ax=x0, ay=y0,
                x=x1, y=y1,
                xref='x', yref='y',
                axref='x', ayref='y',
                showarrow=True,
                arrowhead=3,
                arrowsize=2,
                arrowwidth=2,
                arrowcolor='rgba(100,100,100,0.7)',
                standoff=20,
            )
        )

    # 分别创建seed节点和中间节点的trace
    seed_x, seed_y, seed_text, seed_labels = [], [], [], []
    inter_x, inter_y, inter_text, inter_labels = [], [], [], []

    for node in G.nodes():
        x, y = pos[node]
        attrs = G.nodes[node]
        is_seed = attrs.get('is_seed', False)

        # 计算节点的度数和邻居信息
        degree = G.degree(node)
        neighbors = list(G.neighbors(node))
        predecessors = list(G.predecessors(node))

        # 节点hover信息（包含邻居信息）
        text = f"<b>{attrs['name']}</b><br>"
        text += f"<i>ID: {node}</i><br>"
        text += f"Type: {attrs['node_type']}<br>"
        text += f"Source: {attrs['source']}<br>"
        text += f"Connections: {degree}<br>"
        text += f"<b>{'🔴 Seed Node' if is_seed else '🔵 Intermediate Node'}</b><br>"

        # 添加邻居信息
        if len(neighbors) > 0:
            text += f"<br><b>连接到 ({len(neighbors)}):</b><br>"
            for i, neighbor in enumerate(neighbors[:5]):  # 只显示前5个
                neighbor_name = G.nodes[neighbor]['name']
                text += f"  → {neighbor_name}<br>"
            if len(neighbors) > 5:
                text += f"  ... 还有 {len(neighbors)-5} 个<br>"

        if len(predecessors) > 0:
            text += f"<br><b>来自 ({len(predecessors)}):</b><br>"
            for i, pred in enumerate(predecessors[:5]):
                pred_name = G.nodes[pred]['name']
                text += f"  ← {pred_name}<br>"
            if len(predecessors) > 5:
                text += f"  ... 还有 {len(predecessors)-5} 个<br>"

        # 节点标签（显示名称）
        label = attrs['name']
        if len(label) > 15:
            label = label[:12] + "..."

        if is_seed:
            seed_x.append(x)
            seed_y.append(y)
            seed_text.append(text)
            seed_labels.append(label)
        else:
            inter_x.append(x)
            inter_y.append(y)
            inter_text.append(text)
            inter_labels.append(label)

    # Seed节点trace（更大，带标签）
    seed_trace = go.Scatter(
        x=seed_x, y=seed_y,
        mode='markers+text',
        name='Seed Nodes',
        hoverinfo='text',
        text=seed_labels,
        textposition="top center",
        textfont=dict(size=14, color='#FF6B6B', family='Arial Black'),
        hovertext=seed_text,
        marker=dict(
            color='#FF6B6B',
            size=35,
            line=dict(width=4, color='white'),
            symbol='circle'
        )
    )

    # 中间节点trace（带标签）
    inter_trace = go.Scatter(
        x=inter_x, y=inter_y,
        mode='markers+text',
        name='Intermediate Nodes',
        hoverinfo='text',
        text=inter_labels,
        textposition="top center",
        textfont=dict(size=11, color='#4ECDC4', family='Arial'),
        hovertext=inter_text,
        marker=dict(
            color='#4ECDC4',
            size=25,
            line=dict(width=3, color='white'),
            symbol='circle'
        )
    )

    # 创建图形（添加交互性和动画）
    fig = go.Figure(data=edge_traces + [seed_trace, inter_trace],
                    layout=go.Layout(
                        showlegend=True,
                        hovermode='closest',
                        margin=dict(b=20, l=20, r=20, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=900,
                        plot_bgcolor='rgba(240,242,246,0.5)',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1,
                            font=dict(size=14)
                        ),
                        annotations=edge_annotations,
                        # 添加拖拽和缩放功能
                        dragmode='pan',
                        # 添加动画过渡
                        transition={'duration': 500, 'easing': 'cubic-in-out'}
                    ))

    # 更新配置，启用更多交互功能
    fig.update_layout(
        clickmode='event+select',
        # 启用节点选择
        selectdirection='any'
    )

    return fig

def main():
    # 标题
    st.markdown('<p class="main-header">🔬 MIMIC PrimeKG 子图浏览器</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">交互式探索患者临床知识图谱</p>', unsafe_allow_html=True)

    # 加载节点属性
    with st.spinner("🔄 加载PrimeKG节点属性..."):
        node_attrs = load_node_attributes()

    # 侧边栏
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/medical-heart.png", width=80)
        st.markdown("### 📋 筛选选项")

        task_type = st.selectbox(
            "任务类型",
            ["diagnoses_icd", "prescriptions_atc"],
            help="选择要查看的任务类型"
        )

        # 加载文件列表
        with st.spinner(f"📂 加载文件列表..."):
            df_files = get_file_list(task_type)

        st.metric("📊 总文件数", f"{len(df_files):,}")

        st.markdown("---")
        st.markdown("### 🔍 搜索过滤")

        # 按患者ID筛选
        subject_ids = sorted(df_files['subject_id'].unique())
        selected_subject = st.selectbox(
            "患者ID",
            ["全部"] + list(subject_ids),
            help="选择特定患者或查看全部"
        )

        if selected_subject != "全部":
            df_files = df_files[df_files['subject_id'] == selected_subject]

        st.metric("✅ 筛选后", f"{len(df_files):,} 个文件")

        st.markdown("---")
        st.markdown("### 📖 图例")
        st.markdown("🔴 **Seed节点** - 来自患者EHR")
        st.markdown("🔵 **中间节点** - 来自PrimeKG 3-hop路径")

    # 主界面
    if len(df_files) == 0:
        st.warning("⚠️ 没有找到匹配的文件")
        return

    # 文件选择区域
    st.markdown("### 📁 选择子图文件")

    # 显示文件表格（更美观的样式）
    st.dataframe(
        df_files[['filename', 'subject_id', 'hadm_id', 'context_end']].head(50),
        use_container_width=True,
        height=250
    )

    if len(df_files) > 50:
        st.info(f"ℹ️ 显示前50个文件，共{len(df_files)}个文件")

    # 选择文件
    selected_file = st.selectbox(
        "🎯 选择要可视化的文件",
        df_files['filename'].tolist(),
        help="从列表中选择一个子图文件进行详细查看"
    )

    if selected_file:
        file_path = df_files[df_files['filename'] == selected_file]['path'].iloc[0]

        # 加载子图
        with st.spinner("⏳ 加载子图数据..."):
            data, seed_nodes, all_nodes, edges = load_subgraph(file_path, node_attrs)

        # 基本信息卡片
        st.markdown("---")
        st.markdown("### 📊 子图基本信息")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🆔 Task ID</h3>
                <p style="font-size: 1.2rem;">{data.get('task_id', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>👤 Patient ID</h3>
                <p style="font-size: 1.2rem;">{data.get('subject_id', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>⏰ Context End</h3>
                <p style="font-size: 1.2rem;">{data.get('context_end', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📋 Task Type</h3>
                <p style="font-size: 1.2rem;">{data.get('task', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-box">
            <h4>🎯 预测目标</h4>
            <p>{data.get('target', 'N/A')}</p>
        </div>
        """, unsafe_allow_html=True)

        # 统计信息
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🔴 Seed节点", len(seed_nodes))
        with col2:
            st.metric("📊 总节点", len(all_nodes))
        with col3:
            st.metric("🔵 中间节点", len(all_nodes) - len(seed_nodes))
        with col4:
            st.metric("🔗 边数", len(edges))

        # 可视化
        st.markdown("---")
        st.markdown("### 🌐 交互式网络图")
        st.markdown("💡 *提示: 鼠标悬停查看节点详情，拖拽可以移动视图*")

        with st.spinner("🎨 生成网络图..."):
            fig = create_network_graph(data, seed_nodes, all_nodes, edges, node_attrs)
            st.plotly_chart(fig, use_container_width=True)

        # 节点详情
        st.markdown("---")
        st.markdown("### 📝 节点详细信息")

        tab1, tab2, tab3, tab4 = st.tabs(["🔴 Seed节点", "🔵 中间节点", "🛤️ 路径分析", "📊 统计分析"])

        with tab1:
            seed_data = []
            for node_id in seed_nodes:
                node_str = str(node_id)
                attrs = node_attrs.get(node_str, {})
                node_type = attrs.get('node_type', 'Unknown')
                seed_data.append({
                    'ID': node_id,
                    '名称': attrs.get('name', 'Unknown'),
                    '类型': node_type,
                    '来源': attrs.get('source', 'Unknown')
                })
            df_seed = pd.DataFrame(seed_data)
            st.dataframe(df_seed, use_container_width=True, height=400)
            st.download_button(
                "📥 下载Seed节点数据",
                df_seed.to_csv(index=False).encode('utf-8'),
                f"seed_nodes_{selected_file}.csv",
                "text/csv"
            )

        with tab2:
            intermediate_nodes = [n for n in all_nodes if n not in seed_nodes]
            inter_data = []
            for node_id in intermediate_nodes:
                node_str = str(node_id)
                attrs = node_attrs.get(node_str, {})
                node_type = attrs.get('node_type', 'Unknown')
                inter_data.append({
                    'ID': node_id,
                    '名称': attrs.get('name', 'Unknown'),
                    '类型': node_type,
                    '来源': attrs.get('source', 'Unknown')
                })
            df_inter = pd.DataFrame(inter_data)
            st.dataframe(df_inter, use_container_width=True, height=400)
            st.download_button(
                "📥 下载中间节点数据",
                df_inter.to_csv(index=False).encode('utf-8'),
                f"intermediate_nodes_{selected_file}.csv",
                "text/csv"
            )

        with tab3:
            st.markdown("#### 🛤️ Seed节点之间的连接路径")
            st.markdown("*显示seed节点之间通过中间节点的连接路径*")

            # 分析路径
            with st.spinner("🔍 分析路径..."):
                paths_info = analyze_paths(seed_nodes, all_nodes, edges)

            if len(paths_info) > 0:
                st.success(f"找到 {len(paths_info)} 条seed节点之间的路径")

                # 路径长度分布
                path_lengths = [p['length'] for p in paths_info]
                st.markdown("##### 路径长度分布")
                length_counts = pd.Series(path_lengths).value_counts().sort_index()
                st.bar_chart(length_counts)

                # 选择要可视化的路径
                st.markdown("---")
                st.markdown("##### 🎨 路径可视化")

                # 创建路径选择器
                path_options = []
                for i, path_info in enumerate(paths_info[:20]):
                    src_name = node_attrs.get(str(path_info['source']), {}).get('name', 'Unknown')
                    dst_name = node_attrs.get(str(path_info['target']), {}).get('name', 'Unknown')
                    path_options.append(f"路径 {i+1}: {src_name} → {dst_name} ({path_info['length']}跳)")

                if len(path_options) > 0:
                    selected_path_idx = st.selectbox(
                        "选择要可视化的路径",
                        range(len(path_options)),
                        format_func=lambda x: path_options[x]
                    )

                    # 显示选定路径的详细信息
                    selected_path = paths_info[selected_path_idx]

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        src_name = node_attrs.get(str(selected_path['source']), {}).get('name', 'Unknown')
                        st.metric("起点", src_name)
                    with col2:
                        st.metric("路径长度", f"{selected_path['length']} 跳")
                    with col3:
                        dst_name = node_attrs.get(str(selected_path['target']), {}).get('name', 'Unknown')
                        st.metric("终点", dst_name)

                    # 显示完整路径
                    path_str = " → ".join([
                        node_attrs.get(str(node_id), {}).get('name', f"Node {node_id}")
                        for node_id in selected_path['path']
                    ])
                    st.info(f"**完整路径**: {path_str}")

                    # 生成路径可视化图
                    with st.spinner("🎨 生成路径图..."):
                        path_fig = create_path_visualization(selected_path, all_nodes, edges, node_attrs, seed_nodes)
                        st.plotly_chart(path_fig, use_container_width=True)

                # 显示路径详情表格
                st.markdown("---")
                st.markdown("##### 📋 所有路径列表（前20条）")
                path_data = []
                for i, path_info in enumerate(paths_info[:20]):
                    src_name = node_attrs.get(str(path_info['source']), {}).get('name', 'Unknown')
                    dst_name = node_attrs.get(str(path_info['target']), {}).get('name', 'Unknown')

                    # 构建路径字符串
                    path_str = " → ".join([
                        node_attrs.get(str(node_id), {}).get('name', f"Node {node_id}")
                        for node_id in path_info['path']
                    ])

                    path_data.append({
                        '序号': i + 1,
                        '起点': src_name,
                        '终点': dst_name,
                        '路径长度': path_info['length'],
                        '完整路径': path_str
                    })

                df_paths = pd.DataFrame(path_data)
                st.dataframe(df_paths, use_container_width=True, height=400)

                if len(paths_info) > 20:
                    st.info(f"ℹ️ 显示前20条路径，共{len(paths_info)}条路径")

                st.download_button(
                    "📥 下载路径数据",
                    df_paths.to_csv(index=False).encode('utf-8'),
                    f"paths_{selected_file}.csv",
                    "text/csv"
                )
            else:
                st.warning("⚠️ 未找到seed节点之间的连接路径")

        with tab4:
            # 节点类型统计
            st.markdown("#### 节点类型分布")
            all_types = []
            for node_id in all_nodes:
                node_str = str(node_id)
                attrs = node_attrs.get(node_str, {})
                all_types.append(attrs.get('node_type', 'Unknown'))

            type_counts = pd.Series(all_types).value_counts()
            st.bar_chart(type_counts)

            # 关系类型统计
            st.markdown("#### 关系类型分布")
            rel_types = [edge[2] for edge in edges]
            rel_counts = pd.Series(rel_types).value_counts()
            st.bar_chart(rel_counts)

        # 边详情
        st.markdown("---")
        st.markdown("### 🔗 边详细信息")

        edge_data = []
        for edge in edges[:200]:  # 显示前200条
            src, dst, rel = edge
            src_name = node_attrs.get(str(src), {}).get('name', 'Unknown')
            dst_name = node_attrs.get(str(dst), {}).get('name', 'Unknown')
            edge_data.append({
                '源节点': f"{src_name} ({src})",
                '目标节点': f"{dst_name} ({dst})",
                '关系类型': rel
            })

        df_edges = pd.DataFrame(edge_data)
        st.dataframe(df_edges, use_container_width=True, height=400)

        if len(edges) > 200:
            st.info(f"ℹ️ 显示前200条边，总共{len(edges)}条边")

        st.download_button(
            "📥 下载边数据",
            df_edges.to_csv(index=False).encode('utf-8'),
            f"edges_{selected_file}.csv",
            "text/csv"
        )

if __name__ == '__main__':
    main()
