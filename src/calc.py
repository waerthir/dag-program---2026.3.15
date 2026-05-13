import json
import math
from typing import List, Dict
from dataclasses import dataclass, field
from pathlib import Path

# --- 配置类 (Config) ---
class Config:
    """配置文件，方便后续修改输入路径或目标字段"""

    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    model_keyword = 'qwen2.5-72b'
    mode = 'relationship'

    INPUT_FILE = project_root / 'data' / 'CoT_DAG_compare' / f'output_cot20_{model_keyword}_compare_{mode}_score.json'



    # 如果未来增加了新的评价维度，只需在列表中添加即可
    TARGET_METRICS = ["Fidelity", "Atomicity"]
    TARGET_METRICS = ["Dependency_Accuracy", "Reasoning_Logic_Accuracy", "Reasoning_Type_Accuracy"]
    ROUND_DECIMALS = 4

    metric_dict = {'Fidelity': '信息忠实度',
                   'Atomicity': '原子化程度',
                   "Dependency_Completeness": '依赖关系完整性',
                   "Dependency_Accuracy": '依赖关系准确性',
                   "Reasoning_Logic_Accuracy": '推理逻辑准确性', 
                   "Reasoning_Type_Accuracy": '推理类型准确性'
                   
                   }

# --- 统计核心类 (StatisticsEngine) ---
class StatsEngine:
    """负责具体的数学计算，与业务逻辑解耦"""
    @staticmethod
    def calculate_mean(data: List[float]) -> float:
        return sum(data) / len(data) if data else 0.0

    @staticmethod
    def calculate_variance(data: List[float], mean: float) -> float:
        if len(data) < 2:
            return 0.0
        return sum((x - mean) ** 2 for x in data) / len(data)

# --- 数据处理类 (Processor) ---
class EvaluationProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.metrics_data: Dict[str, List[float]] = {
            metric: [] for metric in self.config.TARGET_METRICS
        }

    def load_and_parse(self, json_data: List[Dict]):
        """解析 JSON 数据并按维度分类存储"""
        for entry in json_data:
            eval_node = entry.get("evaluation", {})
            for metric in self.config.TARGET_METRICS:
                if metric in eval_node:
                    self.metrics_data[metric].append(float(eval_node[metric]))

    def run_statistics(self):
        """执行统计并打印结果"""
        print(f"{'Metric':<15} | {'Mean':<10} | {'Variance':<10}")
        print("-" * 40)
        
        for metric, values in self.metrics_data.items():
            if not values:
                print(f"{metric:<15} | No Data")
                continue
            
            mean = StatsEngine.calculate_mean(values)
            variance = StatsEngine.calculate_variance(values, mean)
            
            print(f"{Config.metric_dict[metric]:<15} | "
                  f"{round(mean, self.config.ROUND_DECIMALS):<10} | "
                  f"{round(variance, self.config.ROUND_DECIMALS):<10}")

# --- 主程序入口 ---
if __name__ == "__main__":
    # 模拟输入数据
    # raw_input = [
    #     {"id": "0_C_1", "evaluation": {"Fidelity": 10, "Atomicity": 5}},
    #     {"id": "0_C_2", "evaluation": {"Fidelity": 8, "Atomicity": 4}},
    #     {"id": "0_C_3", "evaluation": {"Fidelity": 9, "Atomicity": 6}}
    # ]

    # 初始化并运行
    config = Config()
    processor = EvaluationProcessor(config)
    
    # 如果是读取文件：
    with open(config.INPUT_FILE, 'r') as f:
        raw_input = json.load(f)

    print(f'评估数据量是：{len(raw_input)}')
    
    processor.load_and_parse(raw_input)
    processor.run_statistics()