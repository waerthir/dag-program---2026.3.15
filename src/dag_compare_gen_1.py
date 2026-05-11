import json
from pathlib import Path

class Config:

    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent


    model_keyword = "gemma-4-31B-it"
    model_keyword = "llava-v1.6-34b-hf"
    model_keyword = "Qwen3-VL-32B"
    model_keyword = "llava-onevision-72b"
    model_keyword = "nvlm-d-72b"
    model_keyword = "qwen2.5-72b"

    GRAPH_DATA_PATH = project_root / 'data' / 'CoT_o' / f'output_cot20_{model_keyword}_all_graph.json'
    MODEL_DATA_PATH = project_root / 'data' / 'CoT' / f'output_cot20_{model_keyword}.json'
    OUTPUT_PATH = project_root / 'data' / 'CoT_DAG_compare' / f'output_cot20_{model_keyword}_compare_node.json'
    
    # 字段名配置
    ID_SEP = "_"  # 分隔符
    MODEL_KEY = "reasoning_chain_model"
    
    # 节点分类
    CAT_CONDITIONS = "conditions"
    CAT_INTERMEDIATE = "intermediate_steps"
    CAT_FINAL = "final_conclusion"

    ENCODING = "utf-8"





class GraphProcessor:
    def __init__(self, graphs, models):
        self.graphs = graphs
        self.models = models

    def process_all(self):
        result = []
        # 使用 enumerate 确保我们能拿到主 JSON 的索引，从而匹配 models 列表
        for idx, item in enumerate(self.graphs):
            graph_id = item.get("id")
            graph_logic = item.get("graph", {}).get("graph_logic", {})
            
            # 从第二个文件中获取对应的 model 数据
            # 假设 models 是一个 [{model: xx}, {model: xx}] 的列表
            current_model_val = ""
            if idx < len(self.models):
                current_model_val = self.models[idx].get(Config.MODEL_KEY, "")

            # 提取该图下所有的节点
            nodes = self._extract_nodes(graph_id, graph_logic, current_model_val)
            result.extend(nodes)
            
        return result

    def _extract_nodes(self, graph_id, logic, model_val):
        """解析单个图中的所有节点"""
        flattened = []
        
        # 1. 处理 conditions 和 intermediate_steps (列表类型)
        for category in [Config.CAT_CONDITIONS, Config.CAT_INTERMEDIATE]:
            for node in logic.get(category, []):
                flattened.append(self._build_node_dict(graph_id, node, model_val))
        
        # 2. 处理 final_conclusion (字典类型)
        final = logic.get(Config.CAT_FINAL)
        if final:
            flattened.append(self._build_node_dict(graph_id, final, model_val))
            
        return flattened

    def _build_node_dict(self, graph_id, node_data, model_val):
        """统一构建输出格式"""
        return {
            "id": f"{graph_id}{Config.ID_SEP}{node_data.get('id')}",
            "reasoning_chain_model": model_val,
            "content": node_data.get("content", "")
        }
    
    def save_to_file(self, data, file_path=Config.OUTPUT_PATH):
        """
        封装保存逻辑，确保编码正确且格式美观
        """
        try:
            with open(file_path, 'w', encoding=Config.ENCODING) as f:
                # indent=2 让 JSON 具有可读性
                # ensure_ascii=False 确保中文不被转义为 \uXXXX
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"✅ 成功：结果已保存至 {file_path}")
        except Exception as e:
            print(f"❌ 失败：保存文件时出错: {e}")
    

def main():
    # 模拟读取数据
    with open(Config.GRAPH_DATA_PATH, 'r', encoding=Config.ENCODING) as f:
        graphs = json.load(f)

    with open(Config.MODEL_DATA_PATH, 'r', encoding=Config.ENCODING) as f:
        models = json.load(f)
    
    # 实例化并处理
    processor = GraphProcessor(graphs, models)
    final_output = processor.process_all()
    
    # 保存结果
    processor.save_to_file(final_output)

if __name__ == "__main__":
    main()