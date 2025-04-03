import litellm
from litellm import completion
import json
from dotenv import load_dotenv
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import re
import os
import warnings

load_dotenv()

def chinese_font():
    """
    自动检测并返回支持中文的字体
    优先选择中文字体，如果没有则选择支持中文的通用字体
    """
    # 获取所有字体
    fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 中文字体列表（按优先级排序）
    chinese_fonts = [
        'SimHei', 'Microsoft YaHei', 'PingFang SC', 'Hiragino Sans GB',
        'WenQuanYi Micro Hei', 'Source Han Sans CN', 'Noto Sans CJK SC',
        'Droid Sans Fallback', 'Arial Unicode MS'
    ]
    
    # 首先尝试使用中文字体
    for font in chinese_fonts:
        if font in fonts:
            return font
    
    # 如果没有找到中文字体，查找支持中文的通用字体
    for font in fonts:
        # 检查字体是否包含中文字符范围
        try:
            font_path = fm.findfont(font)
            if os.path.exists(font_path):
                with open(font_path, 'rb') as f:
                    font_data = f.read()
                    # 检查是否包含中文字符范围
                    if re.search(b'[\x00-\xff]', font_data) and re.search(b'[\x80-\xff]', font_data):
                        return font
        except:
            continue

    # 如果都没找到，返回默认字体
    warnings.warn("未找到支持中文的字体，使用默认字体")
    return 'DejaVu Sans'

def build_knowledge_graph(text):
    system_prompt = """
    作为知识图谱构建专家，请从文本中提取实体及其关系，并输出以下结构的JSON：
    {
        "abstract": 文本摘要,
        "aspects": 文本的各个角度，即可以从文本的结构来分析，也可以根据内容来判断,
        "reader": 对文本的读者的分析,
        "purpose": 这张图对文本的读者有什么帮助,
        "purposes": [各种具体的帮助],
        "nodes": [{"id": 唯一ID编号, "name": 实体名称, "type": 实体类型}],
        "edges": [{"source": 起点ID, "target": 终点ID, "relation": 关系描述}]
    }
    要求：
    1. 实体类型需简短如[人物/组织/地点/概念]
    2. 关系描述用动宾结构
    3. 确保图结构的连通性
    4. 输出的图结构要对文本的读者有用
    """
    
    try:
        response = completion(
            model="deepseek/deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            response_format={"type": "json_object", "json_schema": {
                "name": "knowledge_graph",
                "schema": {
                    "type": "object",
                    "properties": {
                        "abstract": {"type": "string"},
                        "aspects": {"type": "array", "items": {"type": "string"}},
                        "reader": {"type": "string"},
                        "purpose": {"type": "string"},
                        "purposes": {"type": "array", "items": {"type": "string"}},
                        "nodes": {"type": "array", "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer"},
                                "name": {"type": "string"},
                                "type": {"type": "string"}
                            }
                        }},
                        "edges": {"type": "array", "items": {
                            "type": "object",
                            "properties": {
                                "source": {"type": "integer"},
                                "target": {"type": "integer"},
                                "relation": {"type": "string"}
                            }
                        }}
                    }
                }
            }}
        )
        
        # 提取并解析JSON
        result = response.choices[0].message.content
        return json.loads(result)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def draw_knowledge_graph(graph_data):
    if not graph_data:
        print("No graph data to visualize")
        return
    
    # Set up Chinese font
    font = chinese_font()
    plt.rcParams['font.family'] = font
    plt.rcParams['font.serif'] = [font]
    plt.rcParams['font.sans-serif'] = [font]
    plt.rcParams['font.monospace'] = [font]
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes
    for node in graph_data['nodes']:
        G.add_node(node['id'], name=node['name'], type=node['type'])
    
    # Add edges
    for edge in graph_data['edges']:
        G.add_edge(edge['source'], edge['target'], relation=edge['relation'])
    
    # Set up the plot
    plt.figure(figsize=(15, 10))
    
    # Use spring layout for better node positioning
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw nodes with different colors based on type
    node_colors = []
    for node in G.nodes():
        node_type = G.nodes[node]['type']
        if node_type == '人物':
            node_colors.append('lightblue')
        elif node_type == '组织':
            node_colors.append('lightgreen')
        elif node_type == '地点':
            node_colors.append('lightcoral')
        else:
            node_colors.append('lightgray')
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000)
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True)
    
    # Add labels
    labels = nx.get_node_attributes(G, 'name')
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    # Add edge labels
    edge_labels = nx.get_edge_attributes(G, 'relation')
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6)
    
    plt.title("知识图谱可视化", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# 从文件读取文本
try:
    with open('document.txt', 'r', encoding='utf-8') as file:
        text = file.read()
    graph = build_knowledge_graph(text)
    print(json.dumps(graph, indent=2, ensure_ascii=False))
    draw_knowledge_graph(graph)
except FileNotFoundError:
    print("Error: document.txt not found")
except Exception as e:
    print(f"Error reading file: {str(e)}")
