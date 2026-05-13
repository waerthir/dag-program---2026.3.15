import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict

# --- 配置类 ---
class VisConfig:
    # INPUT_FILE = "data.json"
    # INPUT_FILE = "./CoT_judge_deepseek/output_minicpm/node_score.json"
    # INPUT_FILE = "./CoT_judge_deepseek/output_minicpm/relationship_score.json"
    # INPUT_FILE = "./CoT_judge/output_minicpm/node_score.json"
    # INPUT_FILE = "./CoT_judge/output_minicpm/relationship_score.json"

    # INPUT_FILE = "./CoT_judge_deepseek_reasoner/output_minicpm/node_score.json"
    INPUT_FILE = "./CoT_judge_deepseek_reasoner/output_minicpm/relationship_score.json"


    # OUTPUT_DIR = "plots_output"
    # OUTPUT_DIR = "plots_output_deepseek"
    OUTPUT_DIR = "plots_output_deepseek_reasoner"


    TARGET_METRICS = [
        "Fidelity", "Atomicity", 
        "Dependency_Accuracy",
        "Reasoning_Logic_Accuracy", "Reasoning_Type_Accuracy"
    ]
    # "Dependency_Completeness",
    BIN_COUNT = 11  # 0-10分，刚好11个刻度
    PLOT_STYLE = "whitegrid"
    COLOR_PALETTE = "viridis"

    metric_dict = {'Fidelity': '信息忠实度',
                   'Atomicity': '原子化程度',
                   "Dependency_Completeness": '依赖关系完整性',
                   "Dependency_Accuracy": '依赖关系准确性',
                   "Reasoning_Logic_Accuracy": '推理逻辑准确性', 
                   "Reasoning_Type_Accuracy": '推理类型准确性'
                   
                   }
    


    @staticmethod
    def is_qualified(metric_name: str, score: float) -> bool:
        # if metric_name.lower() == "atomicity":
        #     return score >= 8.0 or score == 5.0
        return score >= 6.0

# --- 绘图引擎类 ---
class VisualizationEngine:
    def __init__(self, config: VisConfig):
        self.config = config

        custom_params = {
            "font.sans-serif": ["SimHei", "Microsoft YaHei", "Arial Unicode MS"],
            "axes.unicode_minus": False
        }
        sns.set_theme(style=self.config.PLOT_STYLE, rc=custom_params)

        if not os.path.exists(self.config.OUTPUT_DIR):
            os.makedirs(self.config.OUTPUT_DIR)

    def draw_distribution(self, metric_name: str, data: List[float]):
        """绘制单个指标的直方图和核密度曲线"""
        if not data:
            print(f"Warning: No data for {metric_name}")
            return

        plt.figure(figsize=(10, 6))
        
        # 绘制直方图和拟合曲线
        sns.histplot(data, bins=self.config.BIN_COUNT, kde=True, 
                     color=sns.color_palette(self.config.COLOR_PALETTE)[3],
                     edgecolor="black")
        
        plt.title(f"Distribution of {metric_name}", fontsize=15)
        plt.xlabel("Score (0-10)", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.xlim(-0.5, 10.5)
        plt.xticks(range(11))

        # 保存图片
        file_path = os.path.join(self.config.OUTPUT_DIR, f"{metric_name}_dist.png")
        plt.savefig(file_path)
        plt.close()
        print(f"Saved: {file_path}")

    def draw_all_distributions(self, extracted_data: Dict[str, List[float]]):
        """将所有指标整合到一张画布中"""
        metrics = [m for m, v in extracted_data.items() if v] # 只处理有数据的指标
        n = len(metrics)
        if n == 0: return

        # 动态计算布局 (例如: 2列布局)
        cols = 2
        rows = (n + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
        axes = axes.flatten() if n > 1 else [axes] # 统一转为列表处理

        for i, metric in enumerate(metrics):
            data = extracted_data[metric]
            ax = axes[i]
            
            # 统计合格数量
            qualified_count = sum(1 for s in data if self.config.is_qualified(metric, s))
            total_count = len(data)
            q_rate = (qualified_count / total_count * 100) if total_count > 0 else 0

            # 绘图
            sns.histplot(data, bins=self.config.BIN_COUNT, kde=True, 
                         ax=ax, color=sns.color_palette(self.config.COLOR_PALETTE)[3])
            
            # 在子图标题中加入统计信息
            ax.set_title(f"{VisConfig.metric_dict[metric]}\n合格率（6分及以上）: {qualified_count}/{total_count} ({q_rate:.1f}%)", fontsize=12)
            ax.set_xlim(-0.5, 10.5)
            ax.set_xticks(range(11))

        # 隐藏多余的子图坐标轴
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        # 根据输入文件名生成输出名，防止覆盖
        save_name = os.path.basename(self.config.INPUT_FILE).replace(".json", "_summary.png")
        plt.savefig(os.path.join(self.config.OUTPUT_DIR, save_name))
        plt.close()

# --- 数据流控制类 ---
class VisualizerApp:
    def __init__(self, config: VisConfig):
        self.config = config
        self.engine = VisualizationEngine(config)

    # def process(self, raw_data: List[Dict]):
    #     # 1. 提取数据
    #     extracted_data = {m: [] for m in self.config.TARGET_METRICS}
    #     for entry in raw_data:
    #         eval_node = entry.get("evaluation", {})
    #         for metric in self.config.TARGET_METRICS:
    #             if metric in eval_node:
    #                 extracted_data[metric].append(float(eval_node[metric]))
        
    #     # 2. 批量绘图
    #     for metric, values in extracted_data.items():
    #         if values:
    #             self.engine.draw_distribution(metric, values)

    def process(self, raw_data: List[Dict]):
        # 1. 提取数据 (逻辑不变)
        extracted_data = {m: [] for m in self.config.TARGET_METRICS}
        for entry in raw_data:
            eval_node = entry.get("evaluation", {})
            for metric in self.config.TARGET_METRICS:
                if metric in eval_node:
                    extracted_data[metric].append(float(eval_node[metric]))
        
        # 2. 一次性生成整合图
        self.engine.draw_all_distributions(extracted_data)

if __name__ == "__main__":
    # 模拟数据输入（实际运行时请替换为读取 JSON 文件）
    with open(VisConfig.INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 示例数据演示
    # sample_data = [
    #     {"evaluation": {"Fidelity": 10, "Atomicity": 1}} for _ in range(50)
    # ] + [
    #     {"evaluation": {"Fidelity": 9, "Atomicity": 10}} for _ in range(50)
    # ]

    app = VisualizerApp(VisConfig())
    app.process(data)