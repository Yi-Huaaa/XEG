#!/usr/bin/env python3
"""Check deterministic CPU construction and measure it at several thread counts.

Run inside a CPU allocation; this script does not request Slurm resources.
Parsing and validation are excluded from the reported construction timings.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--binary', type=Path, default=Path('build/main/cpu'))
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or any(t < 1 for t in args.threads):
        parser.error('threads and repeats must be positive')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    binary, source = args.binary.resolve(), args.input.resolve()
    reference = None
    summary = []
    fields = ['tree_hash', 'best_hash', 'dag_hash', 'filtered_hash', 'tree_cost', 'tree_dag_cost']
    for threads in args.threads:
        report = output / f'threads-{threads}.json'
        command = [str(binary), str(source), str(threads), '0', '1', str(report), str(args.repeats)]
        with (output / f'threads-{threads}.log').open('w') as log:
            subprocess.run(command, cwd=output, stdout=log, stderr=subprocess.STDOUT, check=True)
        rows = json.loads(report.read_text())
        if len(rows) != args.repeats:
            raise RuntimeError('Incomplete construction report')
        for row in rows:
            values = [row[field] for field in fields]
            if reference is None:
                reference = values
            if not row['validated'] or values != reference:
                raise RuntimeError(f'Construction differs at {threads} threads')
        item = {'threads': threads,
                'tree_ms': statistics.median(row['tree_ms'] for row in rows),
                'dag_ms': statistics.median(row['dag_ms'] for row in rows),
                'total_ms': statistics.median(row['tree_ms'] + row['dag_ms'] for row in rows)}
        summary.append(item)
        print(json.dumps(item), flush=True)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
