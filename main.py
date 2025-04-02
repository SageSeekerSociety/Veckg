import litellm
from litellm import completion
import json
from dotenv import load_dotenv

load_dotenv()

def build_knowledge_graph(text):
    system_prompt = """
    作为知识图谱构建专家，请从文本中提取实体及其关系，并输出以下结构的JSON：
    {
        "nodes": [{"id": 唯一标识, "name": 实体名称, "type": 实体类型}],
        "edges": [{"source": 起点ID, "target": 终点ID, "relation": 关系描述}]
    }
    要求：
    1. 实体类型需简短如[人物/组织/地点/概念]
    2. 关系描述用动宾结构
    3. 确保图结构的连通性
    """
    
    try:
        response = completion(

            model="deepseek/deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            response_format={"type": "json_object"}
        )
        
        # 提取并解析JSON
        result = response.choices[0].message.content
        return json.loads(result)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

# 使用示例
text = "苹果公司由史蒂夫·乔布斯创立，总部位于加利福尼亚州。iOS是苹果公司开发的移动操作系统。"
graph = build_knowledge_graph(text)
print(json.dumps(graph, indent=2, ensure_ascii=False))
