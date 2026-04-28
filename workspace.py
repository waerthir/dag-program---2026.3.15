def one_json_process(input_json_name: str):
    json_name = input_json_name
    agent = GraphAgent(Config.API_KEY, Config.MODEL_NAME, Config.SYSTEM_INSTRUCTION)
    
    input_path = GetJsonDir.generate_input_dir(Config.INPUT_DIR, json_name)
    output_path = GetJsonDir.generate_output_dir(Config.OUTPUT_DIR, json_name)
    
    # 1. 预读取已有数据
    all_graphs = FileManager.load_existing_graphs(output_path)
    # 2. 提取已经成功的 ID 集合 (使用 set 提高查询效率)
    completed_ids = {item["id"] for item in all_graphs}
    
    try:
        raw_data = DataProcessor.load_raw_data(input_path)
    except FileNotFoundError:
        logging.error(f"找不到文件: {input_path}")
        return

    # 3. 筛选出未完成的任务
    tasks_to_do = []
    for i, block in enumerate(raw_data):
        if i in completed_ids:
            logging.info(f"任务 [{i}] 已存在，跳过。")
            continue
        tasks_to_do.append((i, block))

    if not tasks_to_do:
        logging.info(f"文件 [{json_name}] 所有任务已完成，无需调用。")
        return

    # 4. 仅处理未完成的任务
    with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
        future_to_id = {
            executor.submit(process_single_task, agent, i, block): i 
            for i, block in tasks_to_do
        }
        
        for future in as_completed(future_to_id):
            result = future.result()
            if result:
                all_graphs.append(result)

    # 5. 重新排序并保存
    all_graphs.sort(key=lambda x: x["id"])
    FileManager.save_all_graphs(all_graphs, output_path)