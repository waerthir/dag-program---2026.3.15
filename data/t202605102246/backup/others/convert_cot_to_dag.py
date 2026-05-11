import json
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
import re

# ================= 配置与日志设置 =================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Config:
    API_KEY = "sk-02edb39c337e4ea3af2d5131b2901d18"
    BASE_URL = "https://api.deepseek.com"
    MODEL_NAME = "deepseek-chat"
    
    JSON_FILES = [
        'output_qwen.json',
        'output_glm.json',
        'output_minicpm.json'
    ]
    INPUT_DIR = r'd:\benchmark\CoT'
    OUTPUT_DIR = r'd:\benchmark\CoT\dag_output'

    MAX_WORKERS = 10  # Adjust based on rate limits

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
5.1.2 文字信息：文本说明（也可能未明确说明）从题目的获取的信息。注意，这类信息一般有其特定的成立条件，可以用此来排除其不属于学科常识，例如“a = 2”
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
'''

# Client setup
client = openai.OpenAI(api_key=Config.API_KEY, base_url=Config.BASE_URL)

def process_text(text):
    messages = [
        {"role": "system", "content": Config.SYSTEM_INSTRUCTION},
        {"role": "user", "content": text}
    ]
    try:
        response = client.chat.completions.create(
            model=Config.MODEL_NAME,
            messages=messages,
            temperature=0.0
        )
        content = response.choices[0].message.content
        
        # Clean up code blocks
        if "```json" in content:
            content = re.split(r"```json", content)[1].split("```")[0].strip()
        elif "```" in content:
            content = re.split(r"```", content)[1].split("```")[0].strip()
        
        # Determine if return is a list or dict, depending on prompt
        # The prompt implies a dag structure. Usually this is a list of nodes or a dict.
        # We'll parse it as is.
        return json.loads(content)
    except Exception as e:
        logging.error(f"Error processing text with DeepSeek: {e}")
        return None

def process_file(filename):
    input_path = os.path.join(Config.INPUT_DIR, filename)
    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        return

    logging.info(f"Reading {filename}...")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to load JSON {filename}: {e}")
        return

    # Use a list to store results in order. Initialize with None
    final_results = [None] * len(data)
    
    with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
        future_map = {}
        for i, item in enumerate(data):
            text = item.get('reasoning_chain_model')
            if text:
                future = executor.submit(process_text, text)
                future_map[future] = i
            else:
                final_results[i] = item # No text to process

        for future in as_completed(future_map):
            index = future_map[future]
            try:
                result = future.result()
                if result:
                    # Merge result into item. 
                    # Assuming result is a dict/list representing the DAG.
                    # We preserve original fields.
                    item_copy = data[index].copy()
                    item_copy['dag'] = result
                    final_results[index] = item_copy
                else:
                    final_results[index] = data[index] # Preserve original on failure
            except Exception as e:
                logging.error(f"Error processing item index {index}: {e}")
                final_results[index] = data[index]

    if not os.path.exists(Config.OUTPUT_DIR):
        os.makedirs(Config.OUTPUT_DIR)
        
    output_filename = filename.replace('.json', '_dag.json')
    output_path = os.path.join(Config.OUTPUT_DIR, output_filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    logging.info(f"Successfully processed {filename} and saved to {output_path}")

if __name__ == "__main__":
    for filename in Config.JSON_FILES:
        process_file(filename)
