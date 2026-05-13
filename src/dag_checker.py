import json
import os
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent

JSON_NAME = [
    'output_cot20_gemma-4-31B-it',
    'output_cot20_llava-onevision-72b',
    'output_cot20_llava-v1.6-34b-hf',
    'output_cot20_nvlm-d-72b',
    'output_cot20_qwen2.5-72b',
    'output_cot20_Qwen3-VL-32B',
    # 'test'
                 ]
# name后面不用加_all_graph，为什么？因为我是啥子，只能写出垃圾的代码

OUTPUT_DIR = project_root / 'data' / 'CoT_o'
# 我不知道为什么要命名为output dir，总之这个其实是input

def check_graph_validity(graph_data):
    """
    检测单个 graph 是否符合要求：
    1. 父节点必须存在 (Conditions + Intermediate + Conclusion)
    2. 必须是 DAG (无环)
    """
    # print('threr is a call')

    logic = graph_data.get("graph_logic", {})
    if not logic:
        return False

    # 1. 收集所有存在的节点 ID
    all_node_ids = set()
    adj = {} # 邻接表: parent -> children
    
    # 获取所有节点定义
    nodes = []
    nodes.extend(logic.get("conditions", []))
    nodes.extend(logic.get("intermediate_steps", []))
    if "final_conclusion" in logic:
        nodes.append(logic.get("final_conclusion"))

    for node in nodes:
        node_id = node.get("id")
        all_node_ids.add(node_id)
        if node_id not in adj:
            adj[node_id] = []

    # 2. 校验父节点存在性并构建边
    # 只有 intermediate_steps 和 final_conclusion 有 parents
    check_nodes = logic.get("intermediate_steps", [])
    if "final_conclusion" in logic:
        check_nodes.append(logic.get("final_conclusion"))

    for node in check_nodes:
        child_id = node.get("id")
        parents = node.get("parents", [])
        for p_id in parents:
            if p_id not in all_node_ids:
                # 违反要求 1：虚空父节点
                return False
            # 构建有向边：Parent -> Child
            adj[p_id].append(child_id)

    # 3. 检测环路 (使用 DFS 染色法)
    # 0: 未访问, 1: 正在访问, 2: 已完成
    visited = {node_id: 0 for node_id in all_node_ids}

    def has_cycle(u):
        visited[u] = 1
        for v in adj.get(u, []):
            if visited[v] == 1:
                return True # 发现回边，有环
            if visited[v] == 0 and has_cycle(v):
                return True
        visited[u] = 2
        return False

    for node_id in all_node_ids:
        if visited[node_id] == 0:
            if has_cycle(node_id):
                # print('huan')
                return False # 违反要求 2：有环

    return True

def validate_json_file(file_path):
    """
    遍历单个 JSON 文件中的所有数据块，打印不合规的 ID
    """
    if not os.path.exists(file_path):
        print(f"警告: 文件不存在 {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    invalid_ids = []
    for item in data_list:
        data_id = item.get("id")
        graph = item.get("graph")
        
        # 如果 graph 是字符串形式，需要解析
        if isinstance(graph, str):
            try:
                graph = json.loads(graph)
            except:
                invalid_ids.append(data_id)
                continue

        if not check_graph_validity(graph):
            invalid_ids.append(data_id)

    if invalid_ids:
        print(f"文件 {os.path.basename(file_path)} 检测到不合规 ID: {invalid_ids}")
    else:
        print(f"文件 {os.path.basename(file_path)} 通过检测。")
    
    return invalid_ids

def batch_process_validation(json_names, output_dir):
    """
    对 json_name 列表进行全局遍历
    """
    print("开始全局图结构合法性检测...")
    print("-" * 40)
    for name in json_names:
        file_path = os.path.join(output_dir, f"{name}_all_graph.json")
        validate_json_file(file_path)
    print("-" * 40)
    print("检测完成。")

# --- 执行脚本 ---
if __name__ == "__main__":


    batch_process_validation(JSON_NAME, OUTPUT_DIR)