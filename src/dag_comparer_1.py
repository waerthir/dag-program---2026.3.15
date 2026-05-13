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
from pathlib import Path
from dotenv import load_dotenv

# ================= 配置与日志设置 =================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Config:

    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent

    JSON_NAME = [

                # 'output_cot20_gemma-4-31B-it_compare_node',
                'output_cot20_llava-v1.6-34b-hf_compare_node',
                # 'output_cot20_Qwen3-VL-32B_compare_node',
                'output_cot20_llava-onevision-72b_compare_node',
                # 'output_cot20_nvlm-d-72b_compare_node',
                # 'output_cot20_qwen2.5-72b_compare_node',
                 ]

    INPUT_DIR = project_root / 'data' / 'CoT_DAG_compare' 
    OUTPUT_DIR = project_root / 'data' / 'CoT_DAG_compare'
    # 这里相当于在inputxx.json后面补上一个xxx_score.json，所以可以不用管，留着看着一样就行

    CHUNKS_DIR = project_root / "data" / "CoT_chunks"
    # 实际会放置在 CHUNKS_DIR / JSON_NAME 下面的数字文档下面
    # 而 json name 又已经做好了区分
    # 所以可以直接使用 CoT_chunks

    load_dotenv()
    API_KEY = os.getenv('GPT-KEY')  # 替换为你的新 Key
    MODEL_NAME = "gpt-5.4"        # 更新模型名称
    BASE_URL = "https://www.msutools.cn/v1" # 或者你使用的代理地址

    MAX_WORKERS = 3

    MAX_RETRIES = 1

    SYSTEM_INSTRUCTION = '''
# Role
你是一位文本对比专家。你的任务是对给定的小节点内容与其出处思维链大文本进行对比和审查，并给出评分

# Rules
评分指标总览
| 信息忠实度: 节点内容是否能在原文中找到出处？是否有模型臆造的信息？
| 原子化程度: 仅仅考虑可能影响推理路径的前提下，节点是否不可再分？
|| 推理路径含义说明: 文本的每个事实和结论可以作为一个节点，节点之间使用推导关系作为边连接。
具体细则
| 信息忠实度：完全能够在原文找到具体、确切的出处，评10分。完全无法在原文找到任何相关联的出处，评0分。
|| 例子：原文为“因为a=1，b=2，所以a+b=3”。对于节点内容“a = 1”，可以完全在节点处找到，评10分；对于节点内容“b = 2”，可以完全在节点处找到，评10分；对于节点内容“c = 3”，完全无法在节点处找到，评0分。
|| 例子：原文为“根据常用的物理常识，苹果会从树上掉下来”。对于节点内容“常用的物理常识万有引力定律”，由于其捏造内容，但是仍然与原文相关，相关程度中等，评5分。
| 原子化程度: 节点完全不可再分，或者是否可再分对整体的推理逻辑路径影响小，评10分。节点可以极其明显地被再分且有再分的必要，评0分。
|| 影响评估：必须拆分的情况：若节点 A 包含信息 A_1 和 A_2，且存在证据表明后续节点 B 仅依赖于 A_1，而节点 C 仅依赖于 A_2，则 A 必须原子化。如果实现原子化，评10分；如果没有实现原子化拆分，评0分。允许合并的情况：若 A_1 和 A_2 总是成对出现作为后续推理的前提，则应保留其语义完整性以减少计算开销。即使没有实现原子化，也评10分。
|| 例子：原文为“因为a=1，b=2，所以a+b=3”。对于节点内容“a = 1”，完全不可再分，评10分；对于节点内容“a = 1，b = 2”，可以明显地看出来可以再分为两个节点“a = 1”和“b = 2”，评0分；
|| 例子：原文为“小明去小红家或者小明去小蓝家，小红家很远，小蓝家很远，所以小明一定要经过很远的距离去他们二人的家里”。对于节点内容“小明去小红家或者小明去小蓝家”，由于是或条件，无法分割为更小的节点，评10分。
|| 例子：原文为“小明乐于助人，待人和善，所以他应该有很多朋友”，对于节点内容“小明乐于助人，待人和善”，其可以被明显再分为“小明乐于助人”和“小明待人和善”，但是由于其在推理关系中只作为一个统一的推理信息单元，因此虽然其没有达成最小化要求，但是不会因此造成推理过程路径的问题，因为其共同导向同一个推理结果，因此评10分。
| 信息忠实度评分写入Fidelity项，原子化程度评分写入Atomicity项。
具体评分尺度
| 所有分数都应该为0到10之间的整数。严禁出现负数，严禁出现大于10的数字
| 评分可以为0到10之间的整数。遵照上面的要求进行打分，不局限于上面给出的例子，结合你自己的思考，上面的例子仅仅作为评分尺度的参考，如果出现判断模糊的情况，依据模糊程度适当给分。

# 输入(输入形式为json)
"reasoning_chain_model": 大文本、原文内容
"content": 小节点内容

# 输出
输出严格按照json格式输出。例子如下：
{
  "Fidelity": 10,
  "Atomicity": 10
}





'''

# ================= 核心处理类 =================

class DataProcessor:
    """数据提取与预处理"""
    @staticmethod
    def load_raw_data(path: str) -> List[List[Dict]]:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    

    @staticmethod
    def format_user_input(block: Dict[str, Any]) -> str:
        """
        直接将原始 block 序列化为 JSON 字符串发送
        """
        # 建议只提取必要的字段，避免传入多余的 id 等信息干扰模型
        input_data = {
            "reasoning_chain_model": block.get("reasoning_chain_model", ""),
            "content": block.get("content", "")
        }
        # 使用 ensure_ascii=False 保证中文正常显示，indent 有助于模型阅读
        return json.dumps(input_data, ensure_ascii=False, indent=2)

class GraphAgent:
    """模型交互"""
    def __init__(self, api_key: str, model: str, prompt: str):
        self.api_key = api_key
        self.model = model
        self.system_prompt = prompt
        self.client = OpenAI(api_key=api_key, base_url=Config.BASE_URL)

    def generate_score(self, user_data: Dict, max_retries: int = 3) -> Optional[str]:
        """生成 DAG，带有网络断线自动重试机制"""

        logging.info('发送信息？')

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
                # response = MultiModalConversation.call(
                response = self.client.chat.completions.create(    
                    model=self.model,
                    # api_key=self.api_key,
                    messages=messages,
                    # result_format='message',
                    temperature=0.1,  # <--- 降低温度，让模型更关注格式而不是发散创造，能轻微提速
                    # enable_context_cache = True,
                    # enable_thinking = False,
                    response_format = {"type": "json_object"},
                    reasoning_effort = "xhigh" 

                )

                # logging.info(f'我消息呢？{response}')

                # if response.status_code == 200:
                    # 获取原始返回内容
                    # raw_content = response.output.choices[0].message.content

                raw_content = response.choices[0].message.content
                
                # 确保返回的一定是纯字符串
                if isinstance(raw_content, list):
                    # 如果是多模态的标准列表格式 [{'text': '...'}]
                    # 将列表里面所有带有 'text' 的部分拼接成完整字符串
                    text_parts =[item.get('text', '') for item in raw_content if isinstance(item, dict)]
                    raw_text = "\n".join(text_parts)
                else:
                    # 如果已经是字符串，直接转存
                    raw_text = str(raw_content)
                    
                return {
                    "text": raw_text,
                    # "usage": response.usage  # dashscope 返回的 usage 字典
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
    def get_chunk_path(json_name: str, task_id: str) -> str:
        """生成碎片文件路径，例如: ./CoT_chunks/output_minicpm/10.json"""
        chunk_dir = os.path.join(Config.CHUNKS_DIR, json_name)
        if not os.path.exists(chunk_dir):
            os.makedirs(chunk_dir, exist_ok=True)
        return os.path.join(chunk_dir, f"{task_id}.json")

    @staticmethod
    def save_single_chunk(data: Dict, json_name: str):
        """将单个任务结果写入独立文件"""
        logging.info(f'成功写入{data["id"]}')
        path = FileManager.get_chunk_path(json_name, data["id"])
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_chunks_ids(json_name: str) -> List[int]:
        """扫描碎片文件夹，获取已完成的 ID 列表"""
        chunk_dir = os.path.join(Config.CHUNKS_DIR, json_name)
        if not os.path.exists(chunk_dir):
            return []
        # 提取文件名中的ID
        return [f.split('.')[0] for f in os.listdir(chunk_dir) if f.endswith('.json')]

    @staticmethod
    def collect_all_chunks(json_name: str) -> List[Dict]:
        """任务结束后，汇总所有碎片数据"""
        chunk_dir = os.path.join(Config.CHUNKS_DIR, json_name)
        results = []
        if os.path.exists(chunk_dir):
            for file_name in sorted(os.listdir(chunk_dir), key=lambda x: x.split('.')[0]):
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
        return os.path.join(output_dir, f'{json_name}_score.json')
    

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
        raw_text = str(data)
        
        # 仅做最终的格式修饰（去空格、分行）
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        return "\n".join(lines)

        

# ================= 主流程 =================

def process_single_task(agent: Any, block: Dict, json_name: str) -> Optional[Dict]:
    # 直接序列化整个 block
    formatted_json_str = DataProcessor.format_user_input(block)
    original_id = block.get("id")
    
    for attempt in range(1, Config.MAX_RETRIES + 1):
        # 这里的输入直接就是 JSON 字符串
        logging.info(f'尝试进行任务{original_id}')
        result_data = agent.generate_score(formatted_json_str, max_retries=1)
        logging.info(f'结果是{result_data}')
        
        if result_data:
            score_json = DataCleaner.clean_response(result_data["text"])
            TokenMonitor.log_usage(original_id, result_data['usage'])
            
            if score_json:
                result = {
                    "id": original_id,
                    "evaluation": score_json
                }
                FileManager.save_single_chunk(result, json_name)
                return result
    return None


def one_json_process(input_json_name: str):
    agent = GraphAgent(Config.API_KEY, Config.MODEL_NAME, Config.SYSTEM_INSTRUCTION)
    input_path = GetJsonDir.generate_input_dir(Config.INPUT_DIR, input_json_name)
    output_path = GetJsonDir.generate_output_dir(Config.OUTPUT_DIR, input_json_name)

    # 从碎片目录恢复进度
    completed_ids = set(FileManager.load_chunks_ids(input_json_name))
    logging.info(f"已发现碎片记录 {len(completed_ids)} 条。")

    try:
        raw_data = DataProcessor.load_raw_data(input_path)
    except FileNotFoundError: return

    all_ids = {str(block.get("id")) for block in raw_data if "id" in block}
    completed_ids = set(FileManager.load_chunks_ids(input_json_name))
    ids_to_do = all_ids - completed_ids
    tasks_to_do = [block for block in raw_data if str(block.get("id")) in ids_to_do]
    logging.info(f"全量任务: {len(all_ids)}，已完成: {len(completed_ids)}，剩余: {len(tasks_to_do)}")

    if not tasks_to_do:
        all_graphs = FileManager.collect_all_chunks(input_json_name)
        FileManager.save_all_graphs(all_graphs, output_path)
        logging.info(f"文件 [{input_json_name}] 已全部完成。")
        return

    # 多线程执行
    with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:

        future_to_id = {
            # 传入 block 即可，不再需要 i
            executor.submit(process_single_task, agent, block, input_json_name): block.get("id")
            for block in tasks_to_do
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
    # print(Config.MODEL_NAME)