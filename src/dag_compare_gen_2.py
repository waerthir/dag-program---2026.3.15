import json
import os
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

    INPUT_GRAPH_PATH = project_root / 'data' / 'CoT_o' / f'output_cot20_{model_keyword}_all_graph.json'
    INPUT_MODEL_PATH = project_root / 'data' / 'CoT' / f'output_cot20_{model_keyword}.json'
    OUTPUT_PATH = project_root / 'data' / 'CoT_DAG_compare' / f'output_cot20_{model_keyword}_compare_relationship.json'

    
    # 编码与格式
    ENCODING = "utf-8"
    INDENT = 2
    
    # 结构字段定义
    CAT_CONDITIONS = "conditions"
    CAT_INTERMEDIATE = "intermediate_steps"
    CAT_FINAL = "final_conclusion"

class GraphProcessor:
    def __init__(self, graphs, models):
        self.graphs = graphs
        self.models = models

    def process(self):
        all_flattened_nodes = []
        
        for idx, item in enumerate(self.graphs):
            graph_id = item.get("id")
            logic = item.get("graph", {}).get("graph_logic", {})
            
            # 获取对应的 model 大文本
            model_content = ""
            if idx < len(self.models):
                model_content = self.models[idx].get("reasoning_chain_model", "")

            # 1. 预处理：建立当前图中 ID 到 Content 的映射表，方便查找 parents
            node_lookup = self._build_node_lookup(logic)
            
            # 2. 提取并转换所有节点
            nodes = self._extract_and_transform(graph_id, logic, model_content, node_lookup)
            all_flattened_nodes.extend(nodes)
            
        return all_flattened_nodes

    def _build_node_lookup(self, logic):
        """建立 ID -> Content 的映射字典"""
        lookup = {}
        # 遍历所有可能的节点来源
        nodes = (
            logic.get(Config.CAT_CONDITIONS, []) + 
            logic.get(Config.CAT_INTERMEDIATE, []) + 
            ([logic.get(Config.CAT_FINAL)] if logic.get(Config.CAT_FINAL) else [])
        )
        for n in nodes:
            if n and "id" in n:
                lookup[n["id"]] = n.get("content", "")
        return lookup

    def _extract_and_transform(self, graph_prefix, logic, model_text, lookup):
        """核心转换逻辑"""
        result = []
        
        # 准备所有节点及其分类标签
        # conditions 通常没有 parents 和 reasoning_logic，但为了结构一致性统一处理
        # 放弃处理
        categories = [
            # (logic.get(Config.CAT_CONDITIONS, []), "条件"),
            (logic.get(Config.CAT_INTERMEDIATE, []), "中间步骤")
        ]
        
        # 处理列表类节点 (Conditions & Intermediate)
        for node_list, _ in categories:
            for node in node_list:
                result.append(self._format_node(graph_prefix, node, model_text, lookup))
        
        # 处理单对象类节点 (Final Conclusion)
        final = logic.get(Config.CAT_FINAL)
        if final:
            result.append(self._format_node(graph_prefix, final, model_text, lookup))
            
        return result

    def _format_node(self, graph_prefix, node, model_text, lookup):
        """将原始节点格式化为目标输出格式"""
        raw_parents = node.get("parents", [])
        
        # 构建 parents 详情列表：[{id: xx, content: xx}]
        parent_details = []
        for p_id in raw_parents:
            parent_details.append({
                "id": f"{p_id}",
                "content": lookup.get(p_id, "未找到父节点内容")
            })

        return {
            "id": f"{graph_prefix}_{node.get('id')}",
            "reasoning_chain_model": model_text,
            "content": node.get("content", ""),
            "parents": parent_details,
            "reasoning_logic": node.get("reasoning_logic", ""),
            "type": node.get("type", "")
        }

    def save(self, data):
        """落盘保存"""
        try:
            with open(Config.OUTPUT_PATH, 'w', encoding=Config.ENCODING) as f:
                json.dump(data, f, indent=Config.INDENT, ensure_ascii=False)
            print(f"🚀 处理完成！共生成 {len(data)} 个节点，已保存至: {Config.OUTPUT_PATH}")
        except Exception as e:
            print(f"❌ 保存失败: {e}")

def main():
    # 检查文件是否存在
    if not os.path.exists(Config.INPUT_GRAPH_PATH) or not os.path.exists(Config.INPUT_MODEL_PATH):
        print("❌ 错误：找不到输入文件，请检查 Config 类中的路径配置。")
        return

    # 读取数据
    with open(Config.INPUT_GRAPH_PATH, 'r', encoding=Config.ENCODING) as f:
        graphs_data = json.load(f)
    with open(Config.INPUT_MODEL_PATH, 'r', encoding=Config.ENCODING) as f:
        models_data = json.load(f)

    # 执行处理
    processor = GraphProcessor(graphs_data, models_data)
    final_data = processor.process()
    processor.save(final_data)

if __name__ == "__main__":
    main()