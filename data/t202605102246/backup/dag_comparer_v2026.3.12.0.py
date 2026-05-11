# 输入1, json：[{问题1}, {问题2}, ...]，
# 输入2，json: [{graph1}, {graph2}, ...]
# 输出：json: [{诊断结果1}, {诊断结果2}, ...]

# 大概思路：输入两份json，分别将数据存入变量CoT_data和graph_data里面
# 创建循环，每个循环都将CoT_data和graph_data的各自的一一对应的数据打包发送给大模型
# 接收大模型的回复，将其保存为json，只需要检查是否是json就可以，类似于上面给出的样式
# 具体参照我发送的代码，模仿里面的代码意图，编写新的代码
# 如果可以封装成类或者函数的地方尽量封装好
# 具体设置全部要放在Config类下面，就像是我发送的代码一样
# 我发送的代码的意图是我给出输入1，大模型给输入2
# system instruction留空，我自己待会填充
# 向模型的发送数据写成给定代码里面给模型的输入1的样式，再粘贴输入2里面的单元素的graph部分
# 虽然和代码本身没什么关系，但是代码里面给大模型的任务是评判两个输入对应地方的匹配度如何。






import json
import os
import re
import logging
from typing import List, Dict, Any, Optional
from dashscope import MultiModalConversation
import base64

# ================= 配置与日志设置 =================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Config:
    API_KEY = "sk-729fa5dcd19e49b4a2db5e8c8402691b" 
    MODEL_NAME = "qwen3.5-plus"
    
    # 输入文件配置
    INPUT_COT_PATH = "biology_reasoning_test_results.json"  # 输入1：包含题目、图片路径和思维链(raw_output)的数据
    INPUT_GRAPH_PATH = "output_test_2/dag.json" # 输入2：之前生成的逻辑图谱数据
    
    # 输出文件配置
    OUTPUT_DIR = "./output_test_2"
    OUTPUT_FILENAME = "dag_compare.json" # 输出：诊断结果
    
    SYSTEM_INSTRUCTION = '''

# Role
你是一个高精度的逻辑图谱审计与评估专家。你的任务是评估一段由原始文本（模型推理过程）转化而来的有向无环图（DAG）是否高质量、忠实地反映了原文本的推理过程。

# Input Data
你将接收到一个 JSON 格式的数据载体（text_payload），包含：
- `resolution`: 原始题目的详细推理过程（Ground Truth 思维链）。
- `graph`: 被评估的图谱 JSON 数据。该图谱由基础证据层（C节点）、中间推理层（I节点）和最终结论层（O节点）构成，其中 I 和 O 节点会通过 `parents` 字段声明其前置依赖。

# Evaluation Criteria (评估标准)

你需要严格执行以下两项核心评估：

## 1. 内容忠实度 (Content Fidelity)
- **幻觉核查 (Hallucination Check)**: 检查图谱中所有节点（尤其是 `content` 和 `reasoning_logic`）是否凭空捏造了 `resolution`中未提及的数值、未使用的定理或外部常识。严禁图谱过度发散。
- **遗漏核查 (Omission Check)**: 仔细阅读 `resolution` 的叙述流，检查是否有关键的中间计算步骤、条件转化或逻辑转折在图谱中被遗漏。

## 2. 推理关系准确性 (Relational Accuracy) - 
针对图谱中的每一个 **I节点（中间步骤）** 和 **O节点（最终结论）**，你必须执行“父节点联立核查”：
- **逻辑闭环判断**: 提取该节点的 `parents` 列表，将被引用的所有父节点内容结合起来。判断这些父节点是否能在逻辑上**充分支撑**该节点的内容生成。
- **原文路径比对**: 这种父子推导路径是否与 `resolution` 原文的思路完全一致？是否存在“张冠李戴”（连错了父节点）、“因果倒置”或“跨步推导”（缺失了某个原本应该作为 parent 的中间 I 节点）？


# Output Format & Content
请深入思考并执行上述核查，最后**仅输出一个合法的 JSON 对象**。不要输出任何 Markdown 代码块标记（如 ```json ）或额外的解释文字。

JSON 输出格式严格要求如下：

{
  "fidelity_diagnosis": {
    "hallucinations":[
      "如果发现幻觉，具体描述哪个节点捏造了什么信息；如果没有，输出空列表 []"
    ],
    "missing_steps":[
      "如果发现图谱遗漏了原文本的关键推理过程，请具体指出；如果没有，输出空列表 []"
    ]
  },
  "relational_accuracy_diagnosis":[
    {
      "node_id": "存在推导关系错误的 I_x 或 O 节点ID",
      "declared_parents":["其声明的父节点ID"],
      "error_description": "详细说明为什么该推导关系不准确（例如：父节点支撑不足、与原文真实推导路径不符、缺少关键前置条件等）。如果所有节点关系均准确，此列表保持为空[]"
    }
  ],
  "metrics": {
    "fidelity_score": <int, 1-10分，10分为完全忠实且无遗漏>,
    "relational_accuracy_score": <int, 1-10分，10分为所有父子连线/推导关系完全符合原文>
  },
  "overall_evaluation": "一句话总结图谱质量，例如：忠实度高但局部推导关系跳跃，总体可用。",
  "is_qualified": <boolean, true或false，综合判断该图谱是否能作为一个合格的思维链图谱使用>
}

'''

# ================= 核心处理类 =================

class DataProcessor:
    """数据提取与预处理"""
    @staticmethod
    def load_json_data(path: str) -> List[Any]:
        """通用的 JSON 加载方法"""
        if not os.path.exists(path):
            logging.error(f"文件不存在: {path}")
            return[]
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
        
    @staticmethod
    def encode_image_to_base64(image_path: str) -> str:
        """将本地图片转换为 base64 编码字符串"""
        if not image_path or not os.path.exists(image_path):
            return ""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    @staticmethod
    def format_evaluation_input(cot_block: Dict[str, Any], graph_block: Dict[str, Any]) -> Dict[str, Any]:
        """
        将原始题目数据(输入1)与图谱数据(输入2)打包为模型需要的输入格式
        """
        # 提取输入1的数据
        picture_path = cot_block.get("image_path", "")
        picture_data = DataProcessor.encode_image_to_base64(picture_path)
        question = cot_block.get("question", "")
        resolution = cot_block.get("raw_output", "")
        
        # 提取输入2的单元素 graph 部分
        # 假设输入2的格式是[{"id": 0, "graph": {...}}, ...]，我们只需要 "graph" 里面的内容
        graph_content = graph_block.get("graph", {})
        
        # 组合成最终发送给模型的输入格式
        return {
            "picture": picture_data, # 这个字段仅用于构建多模态 image url，不放入文本 content 中避免浪费 token
            "text_payload": {
                "question": question,
                "resolution": resolution,
                "graph": graph_content  # 将图谱作为一部分直接粘贴进来
            }
        }

class EvaluationAgent:
    """模型交互（评估/诊断）"""
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.system_prompt = Config.SYSTEM_INSTRUCTION

    def evaluate_match(self, user_data: Dict) -> Optional[str]:
        # 获取需要作为文本发送的 JSON 数据
        text_payload = user_data.get("text_payload", {})
        
        content =[
            {'text': json.dumps(text_payload, ensure_ascii=False)}
        ]
        
        # 如果存在图片，加入多模态 image 节点
        if user_data.get("picture"):
            img_url = f"data:image/png;base64,{user_data['picture']}"
            content.append({'image': img_url})

        messages = [
            {'role': 'system', 'content':[{'text': self.system_prompt}]},
            {'role': 'user', 'content': content}
        ]
        
        # 发送请求
        response = MultiModalConversation.call(
            model=self.model,
            api_key=self.api_key,
            messages=messages
        )

        if response.status_code == 200:
            raw_text = response.output.choices[0].message.content[0]['text']
            return raw_text
        else:
            logging.error(f"API 异常: {response.code} - {response.message}")
            return None

class DataCleaner:
    """数据清洗"""
    @staticmethod
    def clean_response(content: str) -> Optional[Dict]:
        """洗涤回复，去除 Markdown 标签并转为 JSON（检查是否是合法的 JSON）"""
        try:
            # 去除 Markdown 代码块标记
            clean_content = re.sub(r'```(?:json)?\s*|\s*```', '', content).strip()
            # 过滤掉可能的解释性文字，提取大括号包围的内容
            json_match = re.search(r'\{.*\}', clean_content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return None
        except Exception as e:
            logging.error(f"JSON 解析失败: {e}")
            return None

class FileManager:
    """保存结果"""
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def save_all_results(self, all_data: List[Dict[str, Any]], filename: str):
        """
        将所有诊断结果保存到一个 JSON 文件中
        """
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        logging.info(f"所有诊断结果已汇总保存至: {path}")


# ================= 主流程 =================

def main():
    # 初始化各类
    agent = EvaluationAgent(Config.API_KEY, Config.MODEL_NAME)
    file_manager = FileManager(Config.OUTPUT_DIR)
    
    # 读取两份 JSON 文件的数据
    logging.info("正在加载输入数据...")
    cot_data = DataProcessor.load_json_data(Config.INPUT_COT_PATH)
    graph_data = DataProcessor.load_json_data(Config.INPUT_GRAPH_PATH)
    
    if not cot_data or not graph_data:
        logging.error("输入数据为空，请检查文件路径！")
        return

    # 检查两者长度是否一致，避免越界
    if len(cot_data) != len(graph_data):
        logging.warning(f"警告：输入1长度({len(cot_data)}) 与 输入2长度({len(graph_data)}) 不一致！将按最短长度处理。")

    all_diagnosis_results =[]
    
    # 创建循环，一一对应进行打包处理
    total_tasks = min(len(cot_data), len(graph_data))
    for i in range(total_tasks):
        logging.info(f"正在诊断第 {i+1}/{total_tasks} 个任务...")
        
        cot_block = cot_data[i]
        graph_block = graph_data[i]
        
        # 将各自对应的数据打包
        user_input = DataProcessor.format_evaluation_input(cot_block, graph_block)

        # 发送给大模型，获取回复
        raw_result = agent.evaluate_match(user_input)
        
        if raw_result:
            # 检查并清洗返回的 JSON
            cleaned_diagnosis = DataCleaner.clean_response(raw_result)
            if cleaned_diagnosis:
                # 存入汇总列表
                diagnosis_entry = {
                    "id": i,
                    "diagnosis": cleaned_diagnosis
                }
                all_diagnosis_results.append(diagnosis_entry)
                logging.info(f"任务 {i} 诊断完成并已加入队列")
            else:
                logging.error(f"任务 {i} 内容解析失败，返回的内容可能不是标准 JSON 格式。")
        else:
            logging.warning(f"任务 {i} 未生成有效评估结果。")

    # 4. 保存所有结果
    if all_diagnosis_results:
        file_manager.save_all_results(all_diagnosis_results, Config.OUTPUT_FILENAME)
    else:
        logging.warning("没有可保存的诊断结果数据。")
            

if __name__ == "__main__":
    main()