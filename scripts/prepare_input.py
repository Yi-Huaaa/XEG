#!/usr/bin/env python3
"""Restore a missing INPUT.json from INPUT.json.zip without overwriting files."""
import argparse
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import zipfile


def prepare_input(path):
    # Keep the caller's lexical path: symlinks must not change design keys.
    source = Path(os.path.abspath(Path(path).expanduser()))
    if source.name.endswith('.json.zip'):
        source = source.with_suffix('')
    if source.is_file():
        return source
    if source.exists():
        raise ValueError(f'Input is not a regular file: {source}')
    archive = Path(str(source) + '.zip')
    if source.suffix != '.json' or not archive.is_file():
        raise FileNotFoundError(f'Input or matching ZIP not found: {source}')
    temporary = None
    try:
        with zipfile.ZipFile(archive) as packed:
            entries = packed.infolist()
            if (len(entries) != 1 or entries[0].filename != source.name or
                    entries[0].is_dir() or stat.S_ISLNK(entries[0].external_attr >> 16)):
                raise ValueError(f'{archive} must contain only the regular file {source.name}')
            with tempfile.NamedTemporaryFile(dir=source.parent,
                    prefix='.' + source.name + '.', suffix='.part', delete=False) as output:
                temporary = Path(output.name)
                with packed.open(entries[0]) as incoming:
                    shutil.copyfileobj(incoming, output, 1024 * 1024)
            # Publish only after the ZIP CRC has been checked. A concurrent
            # runner may already have published the same complete input.
            temporary.chmod(0o644)
            try:
                os.link(temporary, source)
                print(f'Extracted {archive.name} -> {source}', file=sys.stderr)
            except FileExistsError:
                if not source.is_file():
                    raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return source


def prepare_dataset(directory):
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    inputs = set(directory.rglob('*.json'))
    inputs.update(path.with_suffix('') for path in directory.rglob('*.json.zip'))
    return [prepare_input(path) for path in sorted(inputs) if not path.is_dir()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', nargs='?', type=Path)
    parser.add_argument('--dataset', type=Path, help='Prepare all JSON/ZIP inputs in a directory')
    args = parser.parse_args()
    if bool(args.input) == bool(args.dataset):
        parser.error('Specify one input or --dataset DIRECTORY')
    try:
        paths = prepare_dataset(args.dataset) if args.dataset else [prepare_input(args.input)]
        for path in paths:
            print(path)
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as error:
        parser.exit(1, f'Cannot prepare input: {error}\n')


if __name__ == '__main__':
    main()
