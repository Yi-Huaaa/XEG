import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class RunGpuEnvironmentTest(unittest.TestCase):
    def test_both_wrappers_select_settings_and_forward_overrides(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='xeg-default-profile-') as temporary:
            root = Path(temporary)
            (root / 'build/main').mkdir(parents=True)
            for binary_name in ['gpu', 'simple']:
                binary = root / 'build/main' / binary_name
                binary.write_text('''#!/usr/bin/env python3
import json, pathlib, sys
pathlib.Path('report.json').write_text(json.dumps(sys.argv[1:]))
''')
                binary.chmod(0o755)
            (root / 'input.json').write_text('{}')
            (root / 'custom.json').write_text('{"schema_version": 1}')
            for wrapper in ['run_gpu.sh', 'run.sh']:
                for mode, args in [('default', []), ('geometry', ['2', '128', '1024']),
                                   ('custom', ['--config', 'custom.json']),
                                   ('budget', ['--solver-seconds', '0', '--workers', '1'])]:
                    with self.subTest(wrapper=wrapper, mode=mode):
                        if wrapper == 'run.sh' and mode == 'budget':
                            continue
                        output = root / f'{wrapper}-{mode}'
                        env = dict(os.environ, XEG_BUILD_DIR=str(root / 'build'), XEG_RUN_DIR=str(output))
                        result = subprocess.run([str(repo / 'scripts' / wrapper), 'input.json', *args],
                                                cwd=root, env=env, capture_output=True, text=True)
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        actual = json.loads((output / 'report.json').read_text())
                        self.assertEqual(actual[0], str(root / 'input.json'))
                        self.assertEqual(actual.count('--config'), 1)
                        expected = root / 'custom.json' if mode == 'custom' else repo / 'configs/gpu/87-designs.json'
                        self.assertEqual(actual[actual.index('--config') + 1], str(expected))
                        if mode in ['geometry', 'budget']:
                            self.assertEqual(actual[1:1+len(args)], args)

    def test_wrapper_disables_inherited_binding_with_and_without_solver(self):
        wrapper = Path(__file__).resolve().parents[1] / 'scripts/run_gpu.sh'
        with tempfile.TemporaryDirectory(prefix='xeg-run-gpu-') as temporary:
            root = Path(temporary)
            build = root / 'build'
            binary = build / 'main/gpu'
            binary.parent.mkdir(parents=True)
            binary.write_text('''#!/usr/bin/env python3
import json, os, pathlib, sys
pathlib.Path(sys.argv[sys.argv.index('--report') + 1]).write_text(json.dumps({
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

    def test_relative_config_survives_run_directory_change(self):
        wrapper = Path(__file__).resolve().parents[1] / 'scripts/run_gpu.sh'
        with tempfile.TemporaryDirectory(prefix='xeg-gpu-profile-') as temporary:
            root = Path(temporary)
            binary = root / 'build/main/gpu'
            binary.parent.mkdir(parents=True)
            binary.write_text('''#!/usr/bin/env python3
import json, pathlib, sys
args = sys.argv[1:]
config = pathlib.Path(args[args.index('--config') + 1])
pathlib.Path(args[args.index('--report') + 1]).write_text(json.dumps({
    'args': args, 'config': json.loads(config.read_text())}))
''')
            binary.chmod(0o755)
            (root / 'input.json').write_text('{}')
            (root / 'my config.json').write_text('{"schema_version": 1}')
            output = root / 'run'
            env = dict(os.environ, XEG_BUILD_DIR=str(root / 'build'), XEG_RUN_DIR=str(output))
            result = subprocess.run([str(wrapper), 'input.json', '--config', 'my config.json',
                                     '--design', 'family/input'], cwd=root, env=env,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((output / 'report.json').read_text())
            self.assertEqual(report['args'][:5], [str(root / 'input.json'), '--config',
                             str(root / 'my config.json'), '--design', 'family/input'])
            self.assertEqual(report['config'], {'schema_version': 1})


if __name__ == '__main__':
    unittest.main()
