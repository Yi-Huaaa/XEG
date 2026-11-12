from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import zipfile

from scripts.prepare_input import prepare_input, prepare_dataset


class PrepareInputTest(unittest.TestCase):
    def archive(self, root, name='design.json', entries=None):
        path = root / (name + '.zip')
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as packed:
            for filename, data in (entries or {name: b'{"nodes": {}}'}).items():
                packed.writestr(filename, data)
        return path

    def test_extracts_once_and_preserves_existing_input(self):
        with tempfile.TemporaryDirectory(prefix='xeg-zip-') as directory:
            root = Path(directory)
            archive = self.archive(root, 'input with spaces.json')
            source = prepare_input(archive)
            self.assertEqual(source.read_bytes(), b'{"nodes": {}}')
            source.write_bytes(b'edited by the user')
            self.assertEqual(prepare_input(source).read_bytes(), b'edited by the user')
            self.assertEqual(prepare_input(archive).read_bytes(), b'edited by the user')
            self.assertFalse(list(root.glob('*.part')))

    def test_rejects_nested_extra_and_symlink_members(self):
        with tempfile.TemporaryDirectory(prefix='xeg-zip-') as directory:
            root = Path(directory)
            for entries in [{'../design.json': b'bad'}, {'design.json': b'ok', 'extra': b'bad'}]:
                with self.subTest(entries=list(entries)):
                    self.archive(root, entries=entries)
                    with self.assertRaises(ValueError):
                        prepare_input(root / 'design.json')
                    self.assertFalse((root / 'design.json').exists())
            info = zipfile.ZipInfo('design.json')
            info.external_attr = 0o120777 << 16
            with zipfile.ZipFile(root / 'design.json.zip', 'w') as packed:
                packed.writestr(info, '../target')
            with self.assertRaises(ValueError):
                prepare_input(root / 'design.json')

    def test_bad_crc_leaves_no_partial_input(self):
        with tempfile.TemporaryDirectory(prefix='xeg-zip-') as directory:
            root = Path(directory)
            archive = root / 'design.json.zip'
            with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as packed:
                packed.writestr('design.json', b'unique-payload')
            archive.write_bytes(archive.read_bytes().replace(b'unique-payload', b'broken-payload'))
            with self.assertRaises(zipfile.BadZipFile):
                prepare_input(root / 'design.json')
            self.assertFalse((root / 'design.json').exists())
            self.assertFalse(list(root.glob('*.part')))

    def test_parallel_preparation_and_dataset_deduplication(self):
        with tempfile.TemporaryDirectory(prefix='xeg-zip-') as directory:
            root = Path(directory)
            self.archive(root)
            with ThreadPoolExecutor(max_workers=4) as pool:
                paths = list(pool.map(prepare_input, [root / 'design.json'] * 4))
            self.assertEqual(len(set(paths)), 1)
            self.archive(root, 'second.json')
            self.assertEqual([p.name for p in prepare_dataset(root)], ['design.json', 'second.json'])

    def test_all_runners_restore_input_before_invoking_binary(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='xeg-zip-runner-') as directory:
            root = Path(directory)
            binary_dir = root / 'build/main'
            binary_dir.mkdir(parents=True)
            for name in ['cpu', 'gpu', 'simple']:
                binary = binary_dir / name
                binary.write_text('#!/usr/bin/env python3\nimport pathlib,sys\n'
                                  'assert pathlib.Path(sys.argv[1]).read_bytes() == b\'{}\'\n')
                binary.chmod(0o755)
            for wrapper in ['run_cpu.sh', 'run_gpu.sh', 'run.sh', 'run_all.sh']:
                with self.subTest(wrapper=wrapper):
                    data = root / wrapper / 'inputs'
                    data.mkdir(parents=True)
                    self.archive(data, entries={'design.json': b'{}'})
                    args = ['--dataset', str(data)] if wrapper == 'run_all.sh' else [str(data / 'design.json')]
                    env = dict(os.environ, XEG_BUILD_DIR=str(root / 'build'),
                               XEG_RUN_DIR=str(root / wrapper / 'run'))
                    result = subprocess.run([str(repo / 'scripts' / wrapper), *args], cwd=root,
                                            env=env, text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertEqual((data / 'design.json').read_bytes(), b'{}')


if __name__ == '__main__':
    unittest.main()
