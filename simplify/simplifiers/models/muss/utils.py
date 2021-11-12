# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import bz2
import gzip
import inspect
import os
import shutil
import sys
import tarfile
import tempfile
import time
import zipfile
from contextlib import contextmanager
from io import StringIO
from itertools import zip_longest
from pathlib import Path
from types import MethodType
from urllib.request import urlretrieve

from tqdm import tqdm


def reporthook(count, block_size, total_size):
    # Download progress bar
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size_mb = count * block_size / (1024 * 1024)
    speed = progress_size_mb / duration
    percent = int(count * block_size * 100 / total_size)
    msg = f"\r... {percent}% - {int(progress_size_mb)} MB - {speed:.2f} MB/s - {int(duration)}s"
    sys.stdout.write(msg)


def download(url, destination_path=None, overwrite=True):
    if destination_path is None:
        destination_path = get_temp_filepath()
    if not overwrite and destination_path.exists():
        return destination_path
    print("Downloading...")
    try:
        urlretrieve(url, destination_path, reporthook)
        sys.stdout.write("\n")
    except (Exception, KeyboardInterrupt, SystemExit):
        print("Rolling back: remove partially downloaded file")
        os.remove(destination_path)
        raise
    return destination_path


def download_and_extract(url):
    tmp_dir = Path(tempfile.mkdtemp())
    compressed_filename = url.split("/")[-1]
    compressed_filepath = tmp_dir / compressed_filename
    download(url, compressed_filepath)
    print("Extracting...")
    extracted_paths = extract(compressed_filepath, tmp_dir)
    compressed_filepath.unlink()
    return extracted_paths


def extract(filepath, output_dir):
    output_dir = Path(output_dir)
    # Infer extract function based on extension
    extensions_to_functions = {
        ".tar.gz": untar,
        ".tar.bz2": untar,
        ".tgz": untar,
        ".zip": unzip,
        ".gz": ungzip,
        ".bz2": unbz2,
    }

    def get_extension(filename, extensions):
        possible_extensions = [ext for ext in extensions if filename.endswith(ext)]
        if len(possible_extensions) == 0:
            raise Exception(f"File {filename} has an unknown extension")
        # Take the longest (.tar.gz should take precedence over .gz)
        return max(possible_extensions, key=lambda ext: len(ext))

    filename = os.path.basename(filepath)
    extension = get_extension(filename, list(extensions_to_functions))
    extract_function = extensions_to_functions[extension]

    # Extract files in a temporary dir then move the extracted item back to
    # the ouput dir in order to get the details of what was extracted
    tmp_extract_dir = Path(tempfile.mkdtemp())
    # Extract
    extract_function(filepath, output_dir=tmp_extract_dir)
    extracted_items = os.listdir(tmp_extract_dir)
    output_paths = []
    for name in extracted_items:
        extracted_path = tmp_extract_dir / name
        output_path = output_dir / name
        move_with_overwrite(extracted_path, output_path)
        output_paths.append(output_path)
    return output_paths


def move_with_overwrite(source_path, target_path):
    if os.path.isfile(target_path):
        os.remove(target_path)
    if os.path.isdir(target_path) and os.path.isdir(source_path):
        shutil.rmtree(target_path)
    shutil.move(source_path, target_path)


def untar(compressed_path, output_dir):
    with tarfile.open(compressed_path) as f:
        f.extractall(output_dir)


def unzip(compressed_path, output_dir):
    with zipfile.ZipFile(compressed_path, "r") as f:
        f.extractall(output_dir)


def ungzip(compressed_path, output_dir):
    filename = os.path.basename(compressed_path)
    assert filename.endswith(".gz")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, filename[:-3])
    with gzip.open(compressed_path, "rb") as f_in:
        with open(output_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def unbz2(compressed_path, output_dir):
    extract_filename = os.path.basename(compressed_path).replace(".bz2", "")
    extract_path = os.path.join(output_dir, extract_filename)
    with bz2.BZ2File(compressed_path, "rb") as compressed_file, open(
        extract_path, "wb"
    ) as extract_file:
        for data in tqdm(iter(lambda: compressed_file.read(1024 * 1024), b"")):
            extract_file.write(data)


@contextmanager
def open_files(filepaths, mode="r"):
    files = []
    try:
        files = [Path(filepath).open(mode, encoding="utf-8") for filepath in filepaths]
        yield files
    finally:
        [f.close() for f in files]


def yield_lines_in_parallel(filepaths, strip=True, strict=True, n_lines=float("inf")):
    assert type(filepaths) == list
    with open_files(filepaths) as files:
        for i, parallel_lines in enumerate(zip_longest(*files)):
            if i >= n_lines:
                break
            if None in parallel_lines:
                assert (
                    not strict
                ), f"Files don't have the same number of lines: {filepaths}, use strict=False"
            if strip:
                parallel_lines = [
                    l.rstrip("\n") if l is not None else None for l in parallel_lines
                ]
            yield parallel_lines


class FilesWrapper:
    """Write to multiple open files at the same time"""

    def __init__(self, files, strict=True):
        self.files = files
        self.strict = strict  # Whether to raise an exception when a line is None

    def write(self, lines):
        assert len(lines) == len(self.files)
        for line, f in zip(lines, self.files):
            if line is None:
                assert not self.strict
                continue
            f.write(line.rstrip("\n") + "\n")


@contextmanager
def write_lines_in_parallel(filepaths, strict=True):
    with open_files(filepaths, "w") as files:
        yield FilesWrapper(files, strict=strict)


def write_lines(lines, filepath=None):
    if filepath is None:
        filepath = get_temp_filepath()
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    return filepath


def yield_lines(filepath, gzipped=False, n_lines=None):
    filepath = Path(filepath)
    open_function = open
    if gzipped or filepath.name.endswith(".gz"):
        open_function = gzip.open
    with open_function(filepath, "rt", encoding="utf-8") as f:
        for i, l in enumerate(f):
            if n_lines is not None and i >= n_lines:
                break
            yield l.rstrip("\n")


def read_lines(filepath, gzipped=False):
    return list(yield_lines(filepath, gzipped=gzipped))


def count_lines(filepath):
    n_lines = 0
    # We iterate over the generator to avoid loading the whole file in memory
    for _ in yield_lines(filepath):
        n_lines += 1
    return n_lines


def arg_name_python_to_cli(arg_name, cli_sep="-"):
    arg_name = arg_name.replace("_", cli_sep)
    return f"--{arg_name}"


def kwargs_to_cli_args_list(kwargs, cli_sep="-"):
    cli_args_list = []
    for key, val in kwargs.items():
        key = arg_name_python_to_cli(key, cli_sep)
        if isinstance(val, bool):
            cli_args_list.append(str(key))
        else:
            if isinstance(val, str):
                # Add quotes around val
                assert "'" not in val
                val = f"'{val}'"
            cli_args_list.extend([str(key), str(val)])
    return cli_args_list


def args_dict_to_str(args_dict, cli_sep="-"):
    return " ".join(kwargs_to_cli_args_list(args_dict, cli_sep=cli_sep))


def failsafe_division(a, b, default=0):
    if b == 0:
        return default
    return a / b


@contextmanager
def redirect_streams(source_streams, target_streams):
    # We assign these functions before hand in case a target stream is also a source stream.
    # If it's the case then the write function would be patched leading to infinie recursion
    target_writes = [target_stream.write for target_stream in target_streams]
    target_flushes = [target_stream.flush for target_stream in target_streams]

    def patched_write(self, message):
        for target_write in target_writes:
            target_write(message)

    def patched_flush(self):
        for target_flush in target_flushes:
            target_flush()

    original_source_stream_writes = [
        source_stream.write for source_stream in source_streams
    ]
    original_source_stream_flushes = [
        source_stream.flush for source_stream in source_streams
    ]
    try:
        for source_stream in source_streams:
            source_stream.write = MethodType(patched_write, source_stream)
            source_stream.flush = MethodType(patched_flush, source_stream)
        yield
    finally:
        for (
            source_stream,
            original_source_stream_write,
            original_source_stream_flush,
        ) in zip(
            source_streams,
            original_source_stream_writes,
            original_source_stream_flushes,
        ):
            source_stream.write = original_source_stream_write
            source_stream.flush = original_source_stream_flush


@contextmanager
def mute(mute_stdout=True, mute_stderr=True):
    streams = []
    if mute_stdout:
        streams.append(sys.stdout)
    if mute_stderr:
        streams.append(sys.stderr)
    with redirect_streams(source_streams=streams, target_streams=StringIO()):
        yield


@contextmanager
def log_std_streams(filepath):
    log_file = open(filepath, "w", encoding="utf-8")
    try:
        with redirect_streams(
            source_streams=[sys.stdout], target_streams=[log_file, sys.stdout]
        ):
            with redirect_streams(
                source_streams=[sys.stderr], target_streams=[log_file, sys.stderr]
            ):
                yield
    finally:
        log_file.close()


def add_dicts(*dicts):
    return {k: v for dic in dicts for k, v in dic.items()}


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


TEMP_DIR = None


def get_temp_filepath(create=False):
    global TEMP_DIR
    temp_filepath = Path(tempfile.mkstemp()[1])
    if TEMP_DIR is not None:
        temp_filepath.unlink()
        temp_filepath = TEMP_DIR / temp_filepath.name
        temp_filepath.touch(exist_ok=False)
    if not create:
        temp_filepath.unlink()
    return temp_filepath


def get_temp_dir():
    return Path(tempfile.mkdtemp())


@contextmanager
def create_temp_dir():
    temp_dir = get_temp_dir()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)


@contextmanager
def log_action(action_description):
    start_time = time.time()
    print(f"{action_description}...")
    try:
        yield
    except BaseException as e:
        print(f"{action_description} failed after {time.time() - start_time:.2f}s.")
        raise e
    print(f"{action_description} completed after {time.time() - start_time:.2f}s.")


@contextmanager
def mock_cli_args(args):
    current_args = sys.argv
    sys.argv = sys.argv[:1] + args
    yield
    sys.argv = current_args
