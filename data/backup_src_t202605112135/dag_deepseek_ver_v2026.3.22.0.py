import json
import os
import re
import logging
from typing import List, Dict, Any, Optional
from dashscope import Generation
from dashscope import MultiModalConversation
from concurrent.futures import ThreadPoolExecutor, as_completed  # <--- 新增多线程支持
import time
import base64
import os
import dirtyjson
import codecs
import threading
from openai import OpenAI

# ================= 配置与日志设置 =================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Config:
    API_KEY = "sk-whatapikeys"
    BASE_URL = "https://api.deepseek.com/v1"
    MODEL_NAME = "deepseek-reasoner"
    JSON_NAME = [
                # 'output_glm',
                'output_minicpm',
                # 'output_qwen',
                 ]
    INPUT_DIR = './CoT'
    OUTPUT_DIR = "./CoT_o_deepseek_reasoner"
    CHUNKS_DIR = "./CoT_chunks_deepseek_reasoner"

    MAX_WORKERS = 10

    MAX_RETRIES = 1

    SYSTEM_INSTRUCTION = '''
# Mission
理解之后、我之后会发送题目。在之后的回答，不能做发送json以外的事情

# Role
你是一个高精度的逻辑图谱构建专家。你的核心任务是将题目已知文本信息，即给定的思维链（Chain of Thought），转化为一个严格的有向无环图（DAG）结构。你需要充当“结构化解析者”的角色，将线性的思维链文本拆解为标准化的逻辑节点。
接下来的对话中，请直接回答我的问题，不要展示思考过程，保持回答简洁。


# Constraints & Rules

Part 1: 通用图谱构建原则

1.  极致原子化与拆解
1.1 单一事实协议: 每个 C 节点必须仅包含一个独立的、不可再分的原子事实。
1.1.1 禁止复合句: 严禁出现“A是x，且B是y”或“A和B都...”的结构。必须拆解为：C_1: A是x，C_2: B是y。
1.1.2 数据分离原则: 如果涉及多组数据，禁止压缩在一个节点。不同时间/对象的数值应该拆分为不同节点。
1.2 句法分裂检查: 遇到分号、逗号、或连接词（“并且”、“同时”）这些表示条件并列的语义时，应该在连接处断开，生成多个独立节点。
1.2.1 例子：“已知a=1, b=2, c=3”，应该必须拆分为 C_1, C_2, C_3。
1.3 推理逻辑拆分:
1.3.1 拆分模型: "因为A，且B，所以C" 必须拆解为 -> I_1(确立A) + I_2(确立B) -> I_3(Join A&B 推导C)。
1.3.2 动作分离: 将“引用原理”与“执行计算/对比”分离为不同节点。
1.4 离散条件分离:
1.4.1 枚举检查: 当输入文本使用括号 `(A), (B)`、方括号 `[A], [B]` 或序号 `①...②...` 或逗号 或分号 来罗列多个相互独立的条件时，严禁将其合并在一个节点中。
1.4.2 必须依据这些定界符将内容炸开，生成独立的 C 节点（例如：`C_1: 条件A`, `C_2: 条件B`）。
1.4.3 例外: 如果括号内容表示一个不可分割的数学实体（如坐标点 `(x,y)`、闭区间 `[a,b]` 或化学官能团），则保持其完整性，不进行拆分。

2.  逻辑重构与并行
2.1 隐性步骤显式化: 必须补全输入信息中被间接指出的中间逻辑节点（例如：在生成“对比”节点前，如果提到由事实A，事实B对比而得出节点C，必须先生成“具体事实A”、“具体事实B”这两个前置节点）。
2.2 基于拆解的并行: 当涉及“分析”、“对比”、“综合”时，务必将这些分支独立生成，最后通过 Join 节点汇聚。

3.  合法性自检
3.1 DAG禁止循环引用。所有 I 节点必须有明确的 parents。C 节点无 parents。所有I节点的后继一定能连接上O节点，而不是悬空
3.2 父节点支持: 每一个生成的 I 节点，其 `parents` 应该包括了下列满足条件的节点：文本试图通过这些节点来生成本推理节点。

4.  标准化推理节点类型
4.1 约束: 所有中间推理节点 (I 节点) 和最终结论节点 (O 节点) 的 `type` 字段必须严格属于以下五类之一。注意：原理/常识属于 C 节点内容，不在此列。
4.2 汉字化: 在type这一栏严格使用推理type对应的中文。
4.3 允许种类:
4.3.1 条件转化: 将 Layer 1 的文字/图像证据或学科原理，转化为可操作的数学公式、逻辑表达式或几何关系（例如：由“匀加速”+“牛顿定律”转化出 F=ma）。
4.3.2 逻辑推导: 基于性质、定义或定性关系的推理，不涉及复杂计算（例如：因为 A 是 B 的子集，推导出 A 具有 B 的性质）。
4.3.3 数值计算: 执行具体的数学运算、方程求解、代数变换，产生新的数值结果。（例如，由a=1，b=1推出a+b=2）
4.3.4 对比分析: 对多个节点的结果进行比较、排序，或将计算结果与选项进行核对/排除。这个类型如果被使用，需要检查父节点，确认父节点的正确数量。
4.3.5 综合归纳: 汇聚多个并行的逻辑分支，通过总结或者升华，得出最终陈述。例如，由“中国土地辽阔”“中国物产丰富”归纳出“中国是个大国”

5.  标准化C节点类型
5.1 允许种类：
5.1.1 学科常识：客观成立的事实的引述。例如：“地球绕着太阳进行公转”。
5.1.2 文字信息：文本说明（也可能未明确说明）从题目中获取的信息。注意，这类信息一般有其特定的成立条件，可以用此来排除其不属于学科常识，例如“a = 2”
5.1.3 图像信息：文本说明（也可能未明确说明）从题目的图像中中获取的信息。

6. 各节点内容约束
6.1 C节点: 
6.1.1 仅包含从 resolution 中提取的初始已知条件、模型预设的前提或明确引用的客观常识。
6.1.2 特别注意模型回复的条件来源，例如，模型回复该信息来源于图像，则要标明信息来源于图像；来源于题目则标明来源于题目；来源于学科常识则标明来源于学科常识
6.2 I节点: 
6.2.1 基于前置节点，对 Chain of Thought 进行原子化拆解后的中间步骤。
6.3 O节点: 
6.3.1 最终生成的答案。需要从resolution中提取最终答案作为此节点内容，是resolution推导出的最终终点/最后结论。

7. 分析内容忽略，命题内容保留，推理内容正确归入
7.1 文本中可能含有大量模型表示分析的内容，你需要判断并剔除他们。
7.1.1 例如：“需要分析花鼠对红松种子的传播效果。”中只有“分析”的含义，对于这种文本应该忽略，其内容不应该出现在dag内
7.1.2 例如：“需要分析题目中给出的表格数据”中没有明确的信息，这份信息只指示了后面信息的信息来源
7.1.3 对于这种模型自言自语引导自己分析的内容，不应该放入content内
7.2 文本中大部分内容都是有明确含义的命题信息，你需要判断并保留他们。
7.2.1 例如：“它可能存在于A地或者可能存在与B地”。这是模型提出的命题，应该写入content里面。
7.2.2 例如：“夏洛特的玫瑰图中各个方向的长度分布较为均匀，没有特别突出的方向。”这是模型提出的命题，应该写入content里面。
7.2.3 例如：“无法仅凭玫瑰图信息确定迪拜街道总长度是否最长，因为玫瑰图只展示了各方向的相对频率，而没有提供具体的街道长度数据。”中，“无法仅凭玫瑰图信息确定迪拜街道总长度是否最长”“玫瑰图只展示了各方向的相对频率”“玫瑰图没有提供具体的街道长度数据”这一类属于被确定的命题信息，应该写入content内
7.2.4 例如：“性比的变化可能与花鼠种群的数量波动有关。”这是模型提出的观点。这可能代表了模型的某种推理逻辑的命题来源，这个时候需要写进content里面
7.3 文本中有很多内容属于信息，但是并非起到命题作用，他们应该被放入reasoning_logic里面。


Part 2: 特定输入规则

1.  叙述流拆解
1.1 线性解析: resolution通常是一段连续的文本。你必须按顺序读取，但严禁按句号简单切分节点。必须识别文本中的逻辑语义边界。
1.2 事实与逻辑分离:
1.2.1 当 resolution 陈述“由图可知”、“题目给出”、“根据公式”时，应该提取为 Layer 1 的 C 节点。
1.2.2 当 resolution 陈述“代入计算”、“联立得出”、“对比发现”时，应该转化为 Layer 2+ 的 I 节点。
1.2.3 举例若 resolution 说“因为图中A点坐标(1,2)，代入公式得k=2”，必须拆解为：C_1(图中A点坐标为1,2) -> I_1(引用公式) -> I_2(代入计算得k=2)。

2.  证据提取规范
2.1 隐性证据显式化: resolution 有时会直接使用某个数据而不直接地宣称其来源。
2.1.1 如果 resolution 的推理步骤直接依赖了某个具体的数字或条件（且该条件未在之前生成过 C 节点），必须回溯并补全对应的 C 节点，而且要检查其是否属于学科常识。

3.  逻辑链重构
3.1 过程展开: resolution 往往是人类语言的压缩表达（例如：“联立①②解得 x=5”）。
3.1.1 必须将其“解压缩”。生成：I_a(引用式①), I_b(引用式②), I_c(执行联立求解), I_d(得出x=5)。
3.2 原理处理: 如果 resolution 提到了特定定理或公式名称，必须单独生成一个C节点（内容为引用xxx定理/公式），作为后续计算节点的父节点。

4.  结论映射
4.1 resolution 的最后一句或最终得出的数值结果，对应图谱的汇聚点 O 节点。
4.2 溯源检查: O 节点的 parents 应该指向 resolution 中推导出该结果的最后一批逻辑步骤（I 节点）。

5. 严格文本忠实与源头管控
5.1 思维链锚定协议: 
5.1.1 所有 I 节点的 content 必须严格锚定 resolution 中的原始语义。严禁在转化过程中为了“追求正确”而修正原思维链中的逻辑错误或补充原作者未提及的推导逻辑。
5.2 禁止引入外部进阶推导: 
5.2.1 严禁引入 resolution 文本中未明确出现的学科定理、进阶公式或复杂的外部逻辑链。图谱必须是原始推理过程的“结构化镜像”，而非“知识库扩充”。
5.3 最小化补全边界: 
5.3.1 仅当遇到“计算跳跃”导致 DAG 逻辑断裂时（例如：从 A=1 直接跳到结论，中间缺少 B=2 的过渡），才允许补全最基础的算术运算或等式变形节点。
5.3.2 补全的节点必须标记为最小单元，且不能改变原思维链的推理走向。
5.4 幻觉自检: 
5.4.1 在输出 JSON 前，必须核对每一个 I 节点的 content。如果该事实在 resolution 中无法找到对应的文本支撑，则必须删除该节点或将其合入相邻节点。


# Input Data
- resolution: 模型的完整推理过程文本。这是你构建DAG图谱的唯一信息源。

# Task Definition
请构建一个从“原始信息”到“最终结论”的逻辑推导图。图结构必须满足以下分层要求：

Layer 1: 基础证据层
- ID格式: C_1, C_2, ...
- 约束: 这一层节点没有父节点（没有parent）。

Layer 2: 中间推理层
- ID格式: I_1, I_2, ...
- 逻辑描述: 在 reasoning_logic 中说明该步骤对应思维链的逻辑（例如：“引用公式F=ma...”），不必说出思维链的具体来源。
- 约束: 必须明确列出父节点。

Layer 3: 最终结论层
- ID格式: O
- 约束: 它是 DAG 的汇聚点，由 Chain of Thought 的最后一步得出。

# Output Format (JSON Only)
输出必须严格遵守以下JSON格式，不要包含Markdown代码块标记以外的文本；输出之后，严格检查是否符合json格式；对任何字面反斜杠使用双反斜杠 (例如，“C:\\Users”)；对于多行字符串，在值内使用\n 而不是实际的换行符。：

```json
{
  "graph_logic": {
    "conditions": [
      {
        "id": "C_1",
        "type": "文字条件", 
        "content": "题目中提到......", 
      },
      {
        "id": "C_2",
        "type": "图像条件", 
        "content": "图中.......", 
      },
      {
        "id": "C_3",
        "type": "学科常识", 
        "content": "......", 
      }
    ],
    "intermediate_steps": [
      {
        "id": "I_1",
        "type": "计算",
        "content": "计算出.....",
        "parents": ["C_1", "C_2", "C_3"],
        "reasoning_logic": "根据xxx公式......"
      },
      {
        "id": "I_2",
        "type": "推理",
        "content": "由xxx推理得......",
        "parents": ["I_1", "C_3"],
        "reasoning_logic": "将结果代入......"
      }
    ],
    "final_conclusion": {
      "id": "O",
      "type": "推理",
      "content": ".....",
      "parents": ["I_1", "I_2"],
      "reasoning_logic": "综合......得出最终陈述......"
    }
  }
}
```
'''

# ================= 核心处理类 =================

class DataProcessor:
    """数据提取与预处理"""
    @staticmethod
    def load_raw_data(path: str) -> List[List[Dict]]:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    

    @staticmethod
    def format_user_input(question_block: Dict[str, Any]) -> Dict[str, Any]:
        """
        从字典中提取 CoT，并正确还原转义字符
        """
        resolution = question_block.get("reasoning_chain_model", "")
        
        # 如果 resolution 内部是被包裹在引号里的转义字符串（例如 "line1\\nline2"）
        # 且你发现它没有被正确还原，可以在这里进行“安全二次解析”
        if isinstance(resolution, str) and resolution.startswith('"') and resolution.endswith('"'):
            try:
                # 利用 json 官方解析逻辑，自动处理所有 \\n, \\t, \\"
                resolution = json.loads(resolution)
            except:
                pass 
                
        return {
            "resolution": resolution
        }

class GraphAgent:
    """模型交互"""
    def __init__(self, api_key: str, base_url: str, model: str, prompt: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.system_prompt = prompt

    def generate_dag(self, user_data: Dict, max_retries: int = 3) -> Optional[str]:
        """生成 DAG，带有网络断线自动重试机制"""

        refined_text = ResolutionFormatter.to_plain_text(user_data)
        
        messages =[
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': refined_text}
        ]

        # messages =[
        #     {'role': 'system', 'content': self.system_prompt},
        #     {'role': 'user', 'content': refined_text}
        # ]
        
        for attempt in range(max_retries):
            try:
                # response = Generation.call(
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1,
                    response_format={"type": "json_object"} 
                )

                if response:
                    raw_text = response.choices[0].message.content
                    return {
                        "text": raw_text,
                        "usage": {
                            "input_tokens": response.usage.prompt_tokens,
                            "output_tokens": response.usage.completion_tokens
                        }
                    }
                    
            except Exception as e:
                logging.error(f"请求发生网络或底层异常: {e}")
                if attempt < max_retries - 1:
                    time.sleep(3)
                else:
                    return None

class DataCleaner:
    @staticmethod
    def _extract_json_block(text: str) -> Optional[str]:
        """使用栈平衡原理提取最外层的完整 JSON 块"""

        start = text.find('{')
        if start == -1:
            return None
        
        count = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                count += 1
            elif text[i] == '}':
                count -= 1
                if count == 0:
                    return text[start:i+1]
        
        # 如果没找到闭合，可能被截断了，尝试返回到最后
        return text[start:]

    @staticmethod
    def clean_response(content: str) -> Optional[Dict]:
        """分级降级解析逻辑"""

        # 提取潜在的 JSON 块
        raw_json = DataCleaner._extract_json_block(content)
        if not raw_json:
            logging.error("未能从回复中定位到 JSON 结构")
            return None

        # 标准解析
        try:
            return json.loads(raw_json)
        except json.JSONDecodeError:
            pass

        # 自动闭合处理（处理截断）
        if raw_json.count('{') > raw_json.count('}'):
            temp_content = raw_json + '}' * (raw_json.count('{') - raw_json.count('}'))
            try:
                return json.loads(temp_content)
            except:
                pass

        # 使用更加宽容的 dirtyjson
        try:
            return dirtyjson.loads(raw_json)
        except Exception as e:
            logging.error(f"所有解析尝试均失败。最后一次报错: {e}")
            # 调试：可以将失败的 fixed_content 写入临时文件，方便分析模型到底吐了什么奇怪东西
            return None

class FileManager:
    """保存结果"""

    _write_lock = threading.Lock()

    @staticmethod
    def make_output_dir(output_dir: str):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    @staticmethod
    def save_all_graphs(all_data: List[Dict[str, Any]], path):
        """
        将所有图谱数据一次性保存到一个 JSON 文件中
        """
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        logging.info(f"所有数据已汇总保存至: {path}")

    @staticmethod
    def load_existing_graphs(path: str) -> List[Dict[str, Any]]:
        """尝试读取已有的输出文件，如果不存在或损坏则返回空列表"""
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logging.info(f"成功读取已有进度，共 {len(data)} 条记录。")
                    return data
            except Exception as e:
                logging.warning(f"读取旧文件失败（可能格式损坏），将重新开始: {e}")
        return []
    
    @staticmethod
    def get_chunk_path(json_name: str, task_id: int) -> str:
        """生成碎片文件路径，例如: ./CoT_chunks/output_minicpm/10.json"""
        chunk_dir = os.path.join(Config.CHUNKS_DIR, json_name)
        if not os.path.exists(chunk_dir):
            os.makedirs(chunk_dir, exist_ok=True)
        return os.path.join(chunk_dir, f"{task_id}.json")

    @staticmethod
    def save_single_chunk(data: Dict, json_name: str):
        """将单个任务结果写入独立文件"""
        path = FileManager.get_chunk_path(json_name, data["id"])
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_chunks_ids(json_name: str) -> List[int]:
        """扫描碎片文件夹，获取已完成的 ID 列表"""
        chunk_dir = os.path.join(Config.CHUNKS_DIR, json_name)
        if not os.path.exists(chunk_dir):
            return []
        # 提取文件名中的数字 ID
        return [int(f.split('.')[0]) for f in os.listdir(chunk_dir) if f.endswith('.json')]

    @staticmethod
    def collect_all_chunks(json_name: str) -> List[Dict]:
        """任务结束后，汇总所有碎片数据"""
        chunk_dir = os.path.join(Config.CHUNKS_DIR, json_name)
        results = []
        if os.path.exists(chunk_dir):
            for file_name in sorted(os.listdir(chunk_dir), key=lambda x: int(x.split('.')[0])):
                if file_name.endswith('.json'):
                    with open(os.path.join(chunk_dir, file_name), 'r', encoding='utf-8') as f:
                        results.append(json.load(f))
        return results

class GetJsonDir:
    @staticmethod
    def generate_input_dir(input_dir, json_name):
        return os.path.join(input_dir, f'{json_name}.json')
    
    @staticmethod
    def generate_output_dir(output_dir, json_name):
        return os.path.join(output_dir, f'{json_name}_all_graph.json')
    

class TokenMonitor:
    """Token 消耗监控器"""
    total_input = 0
    total_output = 0

    @classmethod
    def log_usage(cls, task_id: int, usage: Any):
        """记录并打印单次请求的 Token 使用情况"""
        if not usage:
            return
        
        prompt_tokens = usage.get('input_tokens', 0)
        completion_tokens = usage.get('output_tokens', 0)
        
        cls.total_input += prompt_tokens
        cls.total_output += completion_tokens
        
        logging.info(
            f"任务 [{task_id}] Token 统计: "
            f"Input: {prompt_tokens} | Output: {completion_tokens} | "
            f"累计: 输入：{cls.total_input} | 输出：{cls.total_output}"
        )

class ResolutionFormatter:
    """专门负责将复杂的输入数据结构转化为最省 Token 的发送格式"""
    @staticmethod
    def to_plain_text(data: Any) -> str:
        # 提取文本
        if isinstance(data, dict):
            raw_text = str(data.get("resolution", ""))
        else:
            raw_text = str(data)
        
        # 仅做最终的格式修饰（去空格、分行）
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        return "\n".join(lines)

        

# ================= 主流程 =================

def process_single_task(agent: GraphAgent, i: int, block: Dict, json_name: str) -> Optional[Dict]:
    """带有全局重试机制的任务处理函数"""
    user_input = DataProcessor.format_user_input(block)
    
    # 将重试次数上移到此处，覆盖整个处理流
    for attempt in range(1, Config.MAX_RETRIES + 1):
        try:
            logging.info(f"任务 [{i}] 尝试第 {attempt}/{Config.MAX_RETRIES} 次生成...")
            
            # 调用模型生成内容
            result_data = agent.generate_dag(user_input, max_retries=1) # 内部重试设为1，由外部接管
            
            if not result_data:
                logging.warning(f"任务 [{i}] 第 {attempt} 次 API 调用失败，准备重试。")
                continue

            TokenMonitor.log_usage(i, result_data.get("usage"))
            
            # 清洗并解析 JSON
            cleaned_graph = DataCleaner.clean_response(result_data["text"])
            
            if cleaned_graph:
                result = {"id": i, "graph": cleaned_graph}
                FileManager.save_single_chunk(result, json_name)
                logging.info(f"任务 [{i}] 子线程已自动存盘。")
                return result
            else:
                logging.warning(f"任务 [{i}] 第 {attempt} 次 JSON 清洗失败。")
                
            # 如果走到这一步，说明解析或校验失败，触发指数退避（可选，防止 API 频率限制）
            if attempt < Config.MAX_RETRIES:
                time.sleep(1) 

        except Exception as e:
            logging.error(f"任务 [{i}] 第 {attempt} 次执行发生异常: {e}")
            time.sleep(1)

    logging.error(f"任务 [{i}] 在 {Config.MAX_RETRIES} 次重试后仍然失败，已跳过。")
    return None


def one_json_process(input_json_name: str):
    agent = GraphAgent(Config.API_KEY, Config.BASE_URL, Config.MODEL_NAME, Config.SYSTEM_INSTRUCTION)
    input_path = GetJsonDir.generate_input_dir(Config.INPUT_DIR, input_json_name)
    output_path = GetJsonDir.generate_output_dir(Config.OUTPUT_DIR, input_json_name)

    # 从碎片目录恢复进度
    completed_ids = set(FileManager.load_chunks_ids(input_json_name))
    logging.info(f"已发现碎片记录 {len(completed_ids)} 条。")

    try:
        raw_data = DataProcessor.load_raw_data(input_path)
    except FileNotFoundError: return

    tasks_to_do = [(i, block) for i, block in enumerate(raw_data) if i not in completed_ids]
    if not tasks_to_do:
        all_graphs = FileManager.collect_all_chunks(input_json_name)
        FileManager.save_all_graphs(all_graphs, output_path)
        logging.info(f"文件 [{input_json_name}] 已全部完成。")
        return

    # 多线程执行
    with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
        future_to_id = {
            executor.submit(process_single_task, agent, i, block, input_json_name): i 
            for i, block in tasks_to_do
        }

    # 汇总
    logging.info(f"正在汇总 [{input_json_name}] 的所有碎片文件...")
    all_graphs = FileManager.collect_all_chunks(input_json_name)
    FileManager.save_all_graphs(all_graphs, output_path)


def main():
    FileManager.make_output_dir(Config.OUTPUT_DIR)
    for json_name in Config.JSON_NAME:
        logging.info(f"========== 开始处理文件: {json_name} ==========")
        one_json_process(json_name)

if __name__ == "__main__":
    main()