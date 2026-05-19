import argparse
import csv
import json
from pathlib import Path


class Config:
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent

    model_keyword = "gemma-4-31B-it"
    threshold = 8.0
    encoding = "utf-8"

    cot_dir = project_root / "data" / "CoT"
    graph_dir = project_root / "data" / "CoT_o"
    score_dir = project_root / "data" / "CoT_DAG_compare"
    output_dir = project_root / "data" / "CoT_DAG_badcase"


def load_json(path):
    with open(path, "r", encoding=Config.encoding) as f:
        return json.load(f)


def parse_score_id(score_id):
    problem_index, node_id = score_id.split("_", 1)
    return int(problem_index), node_id


def clean_node(node):
    if not node:
        return {}

    result = {
        "id": node.get("id", ""),
        "type": node.get("type", ""),
        "content": node.get("content", ""),
    }
    if "parents" in node:
        result["parents"] = node.get("parents", [])
    if "reasoning_logic" in node:
        result["reasoning_logic"] = node.get("reasoning_logic", "")
    return result


def get_graph_logic(item):
    graph = item.get("graph", {})
    if isinstance(graph, str):
        graph = json.loads(graph)
    return graph.get("graph_logic", {})


def build_graph_index(graph_data):
    graph_index = {}

    for item in graph_data:
        problem_index = item.get("id")
        logic = get_graph_logic(item)
        nodes = {}

        for node in logic.get("conditions", []):
            nodes[node.get("id")] = node
        for node in logic.get("intermediate_steps", []):
            nodes[node.get("id")] = node

        final_node = logic.get("final_conclusion")
        if final_node:
            nodes[final_node.get("id")] = final_node

        graph_index[problem_index] = {
            "graph_logic": logic,
            "nodes": nodes,
        }

    return graph_index


def find_low_metrics(evaluation, threshold):
    low_metrics = {}
    for metric, score in evaluation.items():
        if float(score) < threshold:
            low_metrics[metric] = score
    return low_metrics


def get_cot_item(cot_data, problem_index):
    if 0 <= problem_index < len(cot_data):
        return cot_data[problem_index]
    return {}


def build_base_badcase(case_type, score_item, cot_data, graph_index, threshold):
    evaluation = score_item.get("evaluation", {})
    low_metrics = find_low_metrics(evaluation, threshold)
    if not low_metrics:
        return None

    problem_index, node_id = parse_score_id(score_item.get("id", ""))
    cot_item = get_cot_item(cot_data, problem_index)
    graph_item = graph_index.get(problem_index, {})
    node = graph_item.get("nodes", {}).get(node_id, {})

    return {
        "case_type": case_type,
        "id": score_item.get("id", ""),
        "problem_index": problem_index,
        "problem_id": cot_item.get("problem_id", ""),
        "node_id": node_id,
        "low_metrics": low_metrics,
        "all_scores": evaluation,
        "cot": cot_item.get("reasoning_chain_model", ""),
        "node": clean_node(node),
        "parents": [],
        "graph_context": graph_item.get("graph_logic", {}),
        "review": {
            "status": "",
            "error_type": "",
            "note": "",
            "fix_suggestion": "",
        },
    }


def collect_node_badcases(cot_data, graph_index, node_scores, threshold):
    badcases = []
    for score_item in node_scores:
        badcase = build_base_badcase("node", score_item, cot_data, graph_index, threshold)
        if badcase:
            badcases.append(badcase)
    return badcases


def collect_relationship_badcases(cot_data, graph_index, relationship_scores, threshold):
    badcases = []

    for score_item in relationship_scores:
        badcase = build_base_badcase("relationship", score_item, cot_data, graph_index, threshold)
        if not badcase:
            continue

        graph_item = graph_index.get(badcase["problem_index"], {})
        nodes = graph_item.get("nodes", {})
        parent_ids = badcase.get("node", {}).get("parents", [])
        badcase["parents"] = [clean_node(nodes.get(parent_id, {})) for parent_id in parent_ids]
        badcases.append(badcase)

    return badcases


def write_json(items, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding=Config.encoding) as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def write_csv(items, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_type",
        "id",
        "problem_index",
        "problem_id",
        "node_id",
        "low_metrics",
        "all_scores",
        "node_type",
        "node_content",
        "node_reasoning_logic",
        "parent_ids",
        "parent_contents",
        "cot",
        "review_status",
        "error_type",
        "note",
        "fix_suggestion",
    ]

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            node = item.get("node", {})
            parents = item.get("parents", [])
            writer.writerow({
                "case_type": item.get("case_type", ""),
                "id": item.get("id", ""),
                "problem_index": item.get("problem_index", ""),
                "problem_id": item.get("problem_id", ""),
                "node_id": item.get("node_id", ""),
                "low_metrics": json.dumps(item.get("low_metrics", {}), ensure_ascii=False),
                "all_scores": json.dumps(item.get("all_scores", {}), ensure_ascii=False),
                "node_type": node.get("type", ""),
                "node_content": node.get("content", ""),
                "node_reasoning_logic": node.get("reasoning_logic", ""),
                "parent_ids": "|".join(parent.get("id", "") for parent in parents),
                "parent_contents": "\n".join(parent.get("content", "") for parent in parents),
                "cot": item.get("cot", ""),
                "review_status": "",
                "error_type": "",
                "note": "",
                "fix_suggestion": "",
            })


def build_default_paths(model_keyword):
    return {
        "cot_path": Config.cot_dir / f"output_cot20_{model_keyword}.json",
        "graph_path": Config.graph_dir / f"output_cot20_{model_keyword}_all_graph.json",
        "node_score_path": Config.score_dir / f"output_cot20_{model_keyword}_compare_node_score.json",
        "relationship_score_path": Config.score_dir / f"output_cot20_{model_keyword}_compare_relationship_score.json",
        "json_path": Config.output_dir / f"output_cot20_{model_keyword}_badcase.json",
        "csv_path": Config.output_dir / f"output_cot20_{model_keyword}_badcase.csv",
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_keyword", default=Config.model_keyword)
    parser.add_argument("--threshold", type=float, default=Config.threshold)
    parser.add_argument("--cot_path", type=Path)
    parser.add_argument("--graph_path", type=Path)
    parser.add_argument("--node_score_path", type=Path)
    parser.add_argument("--relationship_score_path", type=Path)
    parser.add_argument("--json_path", type=Path)
    parser.add_argument("--jsonl_path", type=Path)
    parser.add_argument("--csv_path", type=Path)
    parser.add_argument("--no_csv", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    paths = build_default_paths(args.model_keyword)

    cot_path = args.cot_path or paths["cot_path"]
    graph_path = args.graph_path or paths["graph_path"]
    node_score_path = args.node_score_path or paths["node_score_path"]
    relationship_score_path = args.relationship_score_path or paths["relationship_score_path"]
    json_path = args.json_path or args.jsonl_path or paths["json_path"]
    csv_path = args.csv_path or paths["csv_path"]

    cot_data = load_json(cot_path)
    graph_data = load_json(graph_path)
    node_scores = load_json(node_score_path)
    relationship_scores = load_json(relationship_score_path)

    graph_index = build_graph_index(graph_data)
    node_badcases = collect_node_badcases(cot_data, graph_index, node_scores, args.threshold)
    relationship_badcases = collect_relationship_badcases(
        cot_data,
        graph_index,
        relationship_scores,
        args.threshold,
    )
    badcases = node_badcases + relationship_badcases

    write_json(badcases, json_path)
    if not args.no_csv:
        write_csv(badcases, csv_path)

    print(f"node badcase: {len(node_badcases)}")
    print(f"relationship badcase: {len(relationship_badcases)}")
    print(f"total badcase: {len(badcases)}")
    print(f"json saved to: {json_path}")
    if not args.no_csv:
        print(f"csv saved to: {csv_path}")


if __name__ == "__main__":
    main()
