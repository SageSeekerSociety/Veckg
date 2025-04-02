import requests
from bs4 import BeautifulSoup
import spacy
import networkx as nx

def load_nlp_model(language='en'):
    if language == 'en':
        return spacy.load("en_core_web_sm")
    elif language == 'zh':
        return spacy.load("zh_core_web_sm")
    else:
        raise ValueError("Unsupported language")

# 加载中文模型
nlp = load_nlp_model(language='zh')

# 从URL获取网页内容
def get_page_content(url):
    response = requests.get(url)
    
    # 从响应头中获取编码信息
    if 'Content-Type' in response.headers:
        content_type = response.headers['Content-Type'].lower()
        if 'charset=' in content_type:
            charset = content_type.split('charset=')[-1].strip()
            response.encoding = charset
        else:
            # 如果没有指定charset，默认使用utf-8
            response.encoding = 'utf-8'
    else:
        # 如果没有Content-Type头，默认使用utf-8
        response.encoding = 'utf-8'
    
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to retrieve page: {response.status_code}")
        return None

# 使用BeautifulSoup解析网页
def parse_html(content):
    soup = BeautifulSoup(content, "html.parser")
    return soup

# 从网页中提取文本内容
def extract_text_from_html(soup):
    paragraphs = soup.find_all("p")
    text = ""
    for para in paragraphs:
        text += para.get_text()
    return text

# 使用spaCy进行实体识别
def extract_entities(text):
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append(ent.text)
    return entities

# 构建知识图谱
def build_knowledge_graph(entities):
    G = nx.Graph()
    for i, entity1 in enumerate(entities):
        for j, entity2 in enumerate(entities):
            if i != j:  # 避免自我链接
                G.add_edge(entity1, entity2)
    return G

# 可视化知识图谱（可选）
def visualize_graph(G):
    import matplotlib.pyplot as plt
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS 系统
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 设置图形大小
    plt.figure(figsize=(12, 8))
    
    # 绘制图形
    nx.draw(G, with_labels=True, 
            font_weight='bold', 
            node_color='skyblue', 
            node_size=3000, 
            font_size=10,
            font_family='Arial Unicode MS')  # 确保节点标签使用中文字体
    
    plt.show()

# 主程序
def main(url):
    # 获取网页内容
    content = get_page_content(url)
    if not content:
        return

    # 解析HTML并提取文本
    soup = parse_html(content)
    text = extract_text_from_html(soup)

    # 提取实体
    entities = extract_entities(text)
    print(f"Entities extracted: {entities}")

    # 构建知识图谱
    G = build_knowledge_graph(entities)

    # 可视化知识图谱（可选）
    visualize_graph(G)

if __name__ == "__main__":
    # 你可以替换下面的URL为你想抓取的页面
    # 关于组织2025年中国人民大学"求是学术"品牌研究项目立项工作的通知
    url = "http://jiaowu.ruc.edu.cn/xsym/kysjxs/tzggky/fa597e5b3f9e4120bffb9edbbf9c8295.htm#bot"
    main(url)
