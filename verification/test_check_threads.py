import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from verification import check_threads


class CheckThreadsTest(unittest.TestCase):
    def run_checker(self, solutions):
        graph = {'nodes': {'leaf': {'eclass': '0', 'cost': 1, 'children': []}},
                 'root_eclasses': ['0']}
        with tempfile.TemporaryDirectory(prefix='xeg-check-threads-') as temporary:
            root = Path(temporary)
            input_file = root / 'input.json'
            input_file.write_text(json.dumps(graph))
            output = root / 'reports'

            def emit_report(command, **kwargs):
                threads = int(command[2])
                solution = {'selected_enodes': ['leaf'], 'objective': 1,
                            **solutions[threads]}
                report = {key: 'same' for key in
                          ['best_hash', 'tree_hash', 'dag_hash', 'filtered_hash']}
                report.update(threads=threads, solution=solution,
                              solver={'status': 'FEASIBLE', 'wall_time_s': 0.01})
                Path(command[5]).write_text(json.dumps([report]))

            argv = ['check_threads', str(input_file), '--binary', str(root / 'cpu'),
                    '--threads', '1', '2', '--output', str(output)]
            with patch.object(sys, 'argv', argv), \
                    patch.object(check_threads.subprocess, 'run', side_effect=emit_report), \
                    contextlib.redirect_stdout(io.StringIO()):
                check_threads.main()
            return json.loads((output / 'summary.json').read_text())

    def test_matching_solver_sources_without_reason(self):
        rows = self.run_checker({1: {'source': 'solver'}, 2: {'source': 'solver'}})
        self.assertEqual([row['source'] for row in rows], ['solver', 'solver'])
        self.assertEqual([row['reason'] for row in rows], [None, None])
        self.assertTrue(all(row['valid'] and row['objective'] == 1 for row in rows))

    def test_matching_fallback_reasons_are_recorded(self):
        source = {'source': 'tree_fallback', 'reason': 'better_than_solver_incumbent'}
        rows = self.run_checker({1: source, 2: source})
        self.assertTrue(all(row['source'] == 'tree_fallback' and
                            row['reason'] == 'better_than_solver_incumbent' and
                            row['solver_status'] == 'FEASIBLE' for row in rows))

    def test_source_change_rejected_despite_equal_legal_cost_and_hashes(self):
        with self.assertRaisesRegex(RuntimeError, 'source/reason differs at 2 threads'):
            self.run_checker({1: {'source': 'solver'}, 2: {'source': 'tree_fallback',
                              'reason': 'better_than_solver_incumbent'}})

    def test_reason_change_rejected_despite_equal_source_cost_and_hashes(self):
        sources = {1: {'source': 'tree_fallback', 'reason': 'better_than_solver_incumbent'}}
        sources[2] = copy.deepcopy(sources[1])
        sources[2]['reason'] = 'no_solver_incumbent'
        with self.assertRaisesRegex(RuntimeError, 'source/reason differs at 2 threads'):
            self.run_checker(sources)


if __name__ == '__main__':
    unittest.main()
