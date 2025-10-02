import json
import sys
import os
from collections import defaultdict

def fix_file(in_file, out_file):
    with open(in_file, "r") as f:
        data = json.load(f)

    new_nodes = {}
    eclass_to_nodes = defaultdict(list)

    # group nodes by eclass
    for node_id, node in data["nodes"].items():
        eclass = str(node["eclass"])
        eclass_to_nodes[eclass].append(node)

    # assign local indices per eclass
    for eclass, nodes in eclass_to_nodes.items():
        for local_idx, node in enumerate(nodes):
            new_id = f"{eclass}__{local_idx}"
            # convert children to string ids
            fixed_children = []
            for c in node.get("children", []):
                if isinstance(c, int):
                    fixed_children.append(f"{c}__0")  # assume first node in that eclass
                else:
                    fixed_children.append(str(c))
            new_nodes[new_id] = {
                "op": str(node["op"]),
                "cost": float(node.get("cost", 1.0)),
                "eclass": str(node["eclass"]),
                "children": fixed_children,
            }

    new_data = {
        "nodes": new_nodes,
        "root_eclasses": [str(e) for e in data.get("root_eclasses", [])],
    }

    with open(out_file, "w") as f:
        json.dump(new_data, f, indent=2)

def fix_folder(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for fname in os.listdir(input_dir):
        if fname.endswith(".json"):
            in_file = os.path.join(input_dir, fname)
            out_file = os.path.join(output_dir, fname)
            print(f"[INFO] Converting {in_file} -> {out_file}")
            try:
                fix_file(in_file, out_file)
            except Exception as e:
                print(f"[ERROR] Failed to convert {fname}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fix_json.py <input_folder> <output_folder>")
        sys.exit(1)

    fix_folder(sys.argv[1], sys.argv[2])
