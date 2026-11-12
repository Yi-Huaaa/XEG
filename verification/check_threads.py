#!/usr/bin/env python3
"""Check construction, final cost and solution provenance across CPU thread counts."""
import argparse
import json
from pathlib import Path
import subprocess

try:
    from .verify_solution import verify_report
except ImportError:
    from verify_solution import verify_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Input JSON or a directory of input JSON files')
    parser.add_argument('--binary', type=Path, default=Path('build/main/cpu'))
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32])
    parser.add_argument('--deterministic-seconds', type=float, default=1)
    parser.add_argument('--wall-seconds', type=float, default=360)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.threads) < 1 or args.deterministic_seconds <= 0 or args.wall_seconds <= 0:
        parser.error('Thread counts and time limits must be positive')
    source, binary, output = args.input.resolve(), args.binary.resolve(), args.output.resolve()
    inputs = sorted(source.rglob('*.json')) if source.is_dir() else [source]
    if not inputs:
        parser.error('No input JSON files found')
    results = []
    fields = ['best_hash', 'tree_hash', 'dag_hash', 'filtered_hash']
    for path in inputs:
        graph = json.loads(path.read_text())
        relative = path.relative_to(source) if source.is_dir() else Path(path.name)
        directory = output / relative.with_suffix('')
        directory.mkdir(parents=True, exist_ok=True)
        reference = None
        for threads in args.threads:
            report = directory / f'{threads}t.json'
            command = [str(binary), str(path), str(threads), str(args.wall_seconds), '1',
                       str(report), '1', str(args.deterministic_seconds)]
            with (directory / f'{threads}t.log').open('w') as log:
                subprocess.run(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT, check=True)
            rows = json.loads(report.read_text())
            verified = verify_report(graph, rows)[0]
            row = rows[0]
            if row['solver']['wall_time_s'] >= args.wall_seconds * 0.95:
                raise RuntimeError(f'{relative}: wall-time cap may have interrupted deterministic work')
            source = row['solution']['source']
            reason = row['solution'].get('reason')
            signature = [row[k] for k in fields] + [verified['objective'], row['solver']['status'],
                                                   source, reason]
            if reference is None:
                reference = signature
            if signature != reference:
                raise RuntimeError(f'{relative}: construction, final cost, solver status or '
                                   f'solution source/reason differs at {threads} threads')
            result = {'input': str(relative), 'threads': threads, **verified,
                      'solver_status': row['solver']['status'], 'source': source, 'reason': reason}
            results.append(result)
            (output / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
            print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
