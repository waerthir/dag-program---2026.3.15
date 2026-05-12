t202605121721

使用方法维护

dag.py

1，在.env文件里面配置好：QWEN-3.5_APIKEY = sk-xxxx

2，将CoT源文件放置在data / CoT 下面，推荐命名为 output_cot20_{模型名称}.json

3，在 Config 里面的 json_name 内填写好对应的json文件的文件名，比如 output_cot20_gemma-4-31B-it

4，结果会放置在 data / CoT_o 下面，会被自动命名为 output_cot20_{模型名称}_all_graph.json

dag_compare_gen_1.py

1，在命名逻辑符合前面步骤的基础上，修改 Config 内的model_keyword为先前的对应的{模型名称}

2，生成的文件会在 data / CoT_DAG_compare 内，命名为 output_cot20_{模型名称}_compare_node

3，本脚本会将图文件拆分为对应的CoT-节点对，使得能够进行后续的评估

dag_compare_gen_2.py

1，在命名逻辑符合前面步骤的基础上，修改 Config 内的model_keyword为先前的对应的{模型名称}

2，生成的文件会在 data / CoT_DAG_compare 内，命名为 output_cot20_{模型名称}_compare_relationship

3，本脚本会将图文件拆分为对应的CoT-节点关系对，使得能够进行后续的评估

dag_compare_1.py

1，在.env文件里面配置好：GPT-KEY = sk-xxxx

2，确保你做好了上面提到的所有步骤，然后在config 的 json name里面填入你在dag_compare_gen_1.py得到的对应的CoT-节点对json文件名称，比如'output_cot20_Qwen3-VL-32B_compare_node'

3，结果会放置在 data / CoT_DAG_compare 下面，会被自动命名为 output_cot20_{模型名称}_compare_node_score.json

dag_compare_2.py

1，在.env文件里面配置好：GPT-KEY = sk-xxxx

2，确保你做好了上面提到的所有步骤，然后在config 的 json name里面填入你在dag_compare_gen_2.py得到的对应的CoT-节点关系对json文件名称，比如'output_cot20_Qwen3-VL-32B_compare_relationship'

3，结果会放置在 data / CoT_DAG_compare 下面，会被自动命名为 output_cot20_{模型名称}_compare_relationship_score.json




t202605121732

两个calc脚本用于统计对应的数据



