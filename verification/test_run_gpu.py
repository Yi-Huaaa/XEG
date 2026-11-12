import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class RunGpuEnvironmentTest(unittest.TestCase):
    def test_wrapper_disables_inherited_binding_with_and_without_solver(self):
        wrapper = Path(__file__).resolve().parents[1] / 'scripts/run_gpu.sh'
        with tempfile.TemporaryDirectory(prefix='xeg-run-gpu-') as temporary:
            root = Path(temporary)
            build = root / 'build'
            binary = build / 'main/gpu'
            binary.parent.mkdir(parents=True)
            binary.write_text('''#!/usr/bin/env python3
import json, os, pathlib, sys
pathlib.Path(sys.argv[7]).write_text(json.dumps({
    'bind': os.environ.get('OMP_PROC_BIND'),
    'places': os.environ.get('OMP_PLACES'),
    'arguments': sys.argv[1:7]}))
''')
            binary.chmod(0o755)
            source = root / 'input with spaces.json'
            source.write_text('{}')
            for seconds in ['0', '360']:
                with self.subTest(seconds=seconds):
                    output = root / f'run-{seconds}'
                    env = dict(os.environ, OMP_PROC_BIND='close', OMP_PLACES='cores',
                               XEG_BUILD_DIR=str(build), XEG_RUN_DIR=str(output))
                    result = subprocess.run([str(wrapper), str(source), '2', '64', '512', seconds, '8'],
                                            env=env, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    report = json.loads((output / 'report.json').read_text())
                    self.assertEqual(report['bind'], 'false')
                    self.assertIsNone(report['places'])
                    self.assertEqual(report['arguments'], [str(source), '2', '64', '512', seconds, '8'])
                    self.assertTrue((output / 'run.log').is_file())


if __name__ == '__main__':
    unittest.main()
