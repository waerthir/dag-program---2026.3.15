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

dag_badcase.py

src/dag_badcase.py 现在可用参数如下。

--model_keyword：模型名。默认 gemma-4-31B-it。只有在文件名符合默认格式时才需要用它。

--threshold：低分阈值。默认 8.0。脚本筛选规则是“分数 < threshold”，不包含等于。

--cot_path：CoT 原文件路径。对应 data/CoT 下的文件。

--graph_path：graph 文件路径。对应 data/CoT_o 下的文件。

--node_score_path：节点评分文件路径。对应 xxx_node_score.json。

--relationship_score_path：关系评分文件路径。对应 xxx_relationship_score.json。

--jsonl_path：输出 jsonl 路径。jsonl 保留信息最完整，包含 cot、node、parents、graph_context、review 字段。

--csv_path：输出 csv 路径。适合表格人工审核，内容比 jsonl 扁平一些。

--no_csv：只输出 jsonl，不输出 csv。

例如：

```powershell
E:\TrashE\Miniconda3\envs\dag_env\python.exe src\dag_badcase.py `
--cot_path                  data\CoT\output_cot20_llava-onevision-72b.json `
--graph_path                data\CoT_o\output_cot20_llava-onevision-72b_all_graph.json `
--node_score_path           data\CoT_DAG_compare\output_cot20_llava-onevision-72b_compare_node_score.json `
--relationship_score_path   data\CoT_DAG_compare\output_cot20_llava-onevision-72b_compare_relationship_score.json `
--jsonl_path                data\CoT_DAG_badcase\output_cot20_llava-onevision-72b_badcase_threshold5.jsonl `
--csv_path                  data\CoT_DAG_badcase\output_cot20_llava-onevision-72b_badcase_threshold5.csv `
--threshold 5
```

calc.py

1，在满足前面命名规则的基础上，修改config里面的model_keyword和mode，会在命令行窗口输出对应的均值和方差

2，还要注意node模式和relationship模式需要选用不同的target metric

calc_histogram.py

1，在满足前面命名规则的基础上，修改config里面的model_keyword和mode，会在 data / plot输出对应的图像

2，is_qualified 方法里面可以修改threshold。



