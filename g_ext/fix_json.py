import json
import sys
import os

def fix_file(in_file, out_file):
    with open(in_file, "r") as f:
        data = json.load(f)

    new_nodes = {}
    for node_id, node in data["nodes"].items():
        new_node = dict(node)

        # eclass → string
        if "eclass" in new_node and not isinstance(new_node["eclass"], str):
            new_node["eclass"] = str(new_node["eclass"])

        # children → string list
        if "children" in new_node:
            fixed_children = []
            for c in new_node["children"]:
                if not isinstance(c, str):
                    fixed_children.append(str(c))
                else:
                    fixed_children.append(c)
            new_node["children"] = fixed_children

        new_nodes[node_id] = new_node

    # root_eclasses 也可能需要轉 string
    root_eclasses = []
    for r in data.get("root_eclasses", []):
        if not isinstance(r, str):
            root_eclasses.append(str(r))
        else:
            root_eclasses.append(r)

    new_data = {
        "nodes": new_nodes,
        "root_eclasses": root_eclasses,
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
            fix_file(in_file, out_file)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fixed_json.py <input_folder> <output_folder>")
        sys.exit(1)

    fix_folder(sys.argv[1], sys.argv[2])
