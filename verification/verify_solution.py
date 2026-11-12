#!/usr/bin/env python3
"""Validate exported CPU or GPU solutions against the original input JSON, without XEG."""
import argparse
from collections import deque
import json
import math
from pathlib import Path
import struct


def verify(graph, solution):
    nodes = graph['nodes']
    selected = solution['selected_enodes']
    if len(set(selected)) != len(selected):
        raise ValueError('Duplicate selected e-node')
    choices = {}
    for name in selected:
        if name not in nodes:
            raise ValueError(f'Unknown e-node {name}')
        cls = str(nodes[name]['eclass'])
        if cls in choices:
            raise ValueError(f'Multiple choices for e-class {cls}')
        choices[cls] = name
    roots = {str(c) for c in graph['root_eclasses']}
    if not roots.issubset(choices):
        raise ValueError(f'Missing roots: {sorted(roots - choices.keys())}')
    edges = {}
    objective = 0
    for cls, name in choices.items():
        children = set()
        for child in nodes[name]['children']:
            if isinstance(child, str):
                if child not in nodes:
                    raise ValueError(f'Unknown child e-node {child}')
                target = str(nodes[child]['eclass'])
            elif isinstance(child, int) and not isinstance(child, bool):
                target = str(child)
            else:
                raise ValueError(f'Invalid child reference {child!r}')
            if target not in choices:
                raise ValueError(f'Selected e-node {name} has missing child e-class {target}')
            children.add(target)
        edges[cls] = children
        # Match the documented reader (float32) and CP-SAT integer objective.
        cost = struct.unpack('f', struct.pack('f', nodes[name]['cost']))[0]
        if not math.isfinite(cost) or not 0 <= cost <= 2147483647:
            raise ValueError(f'Invalid CP-SAT cost on {name}')
        objective += int(cost)
    indegree = dict.fromkeys(choices, 0)
    for children in edges.values():
        for child in children:
            indegree[child] += 1
    queue = deque(c for c, degree in indegree.items() if degree == 0)
    visited = 0
    while queue:
        cls = queue.popleft()
        visited += 1
        for child in edges[cls]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if visited != len(choices):
        raise ValueError('Selected e-class graph contains a cycle')
    if objective != solution['objective']:
        raise ValueError(f'Objective mismatch: evaluated {objective}, reported {solution["objective"]}')
    reachable = set(roots)
    queue = deque(roots)
    while queue:
        for child in edges[queue.popleft()]:
            if child not in reachable:
                reachable.add(child)
                queue.append(child)
    return {'valid': True, 'objective': objective, 'selected_classes': len(choices),
            'root_classes': len(roots), 'unreachable_selected_classes': len(choices) - len(reachable)}


def verify_report(graph, reports):
    results = []
    for report in reports:
        solution = report.get('solution', report.get('solver', {}))
        if 'selected_enodes' not in solution or 'objective' not in solution:
            raise ValueError('Report does not contain a feasible solution')
        result = verify(graph, solution)
        result.update(threads=report.get('threads'), repeat=report.get('repeat'),
                      source=solution.get('source', 'solver'))
        results.append(result)
    if not results:
        raise ValueError('Empty report')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = verify_report(json.loads(args.input.read_text()), json.loads(args.report.read_text()))
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
