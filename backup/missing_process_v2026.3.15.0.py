import json
import os

JSON_NAMES = [
    # 'output_glm_physics',
    # 'output_minicpm_physics',
    # 'output_glm_math', 
    # 'output_minicpm_math',
    # 'output_glm_biology', 
    # 'output_minicpm_biology',
    # 'output_qwen_physics',
    # 'output_qwen_math',
    # 'output_qwen_biology',
    # 'output_qwen_biology2',
    'output_glm', 

]
INPUT_DIR = './CoT'
OUTPUT_DIR = "./CoT_o"
RECOVERY_DIR = "./CoT_m" # 新增：存放提取出的缺失数据


def process_missing_data(input_json_path, output_json_path, recovery_path):
    # 1. 加载数据（input_data 为完整列表）
    with open(input_json_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    with open(output_json_path, 'r', encoding='utf-8') as f:
        output_data = json.load(f)

    # 2. 获取输出文件中现存的索引集合
    # 假设 output 里的 id 对应的是 input 列表的原始下标 (0, 1, 2...)
    output_existing_indices = {int(item['id']) for item in output_data}

    # 3. 找出缺失的索引
    total_count = len(input_data)
    all_indices = set(range(total_count))
    missing_indices = sorted(list(all_indices - output_existing_indices))

    # 4. 统计信息
    missing_count = len(missing_indices)
    missing_rate = (missing_count / total_count) * 100 if total_count > 0 else 0
    
    # 打印统计结果...（逻辑不变）
    print(f'missing count is: {missing_count}')
    print(f'missing rate is: {missing_rate}')
    print(f'missing indices is: {missing_indices}')

    # 5. 根据索引从 input_data 列表中提取原始数据块
    missing_blocks = [input_data[idx] for idx in missing_indices]
    
    if missing_blocks:
        with open(recovery_path, 'w', encoding='utf-8') as f:
            json.dump(missing_blocks, f, ensure_ascii=False, indent=4)
        print(f"  - 缺失数据已保存至: {recovery_path}")
    else:
        print("  - 无缺失数据，无需输出文件。")
    print("-" * 40)

def main():
    if not os.path.exists(RECOVERY_DIR):
        os.makedirs(RECOVERY_DIR)

    for name in JSON_NAMES:
        # 拼接路径：input 里面通常是原始名，output 里面是带 output_ 前缀的名
        # 这里假设 input 文件名为 name.json (或根据你实际情况调整)
        # 如果 input 里的文件名不带 output_，可以做个 strip
        
        input_path = os.path.join(INPUT_DIR, f"{name}.json")
        output_path = os.path.join(OUTPUT_DIR, f"{name}_all_graph.json")
        recovery_path = os.path.join(RECOVERY_DIR, f"missing_{name}.json")

        if os.path.exists(input_path) and os.path.exists(output_path):
            process_missing_data(input_path, output_path, recovery_path)
        else:
            print(f"跳过: 文件未找到 ({input_path} 或 {output_path})")

if __name__ == "__main__":
    main()