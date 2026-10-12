"""Read-only disjoint SSH ranges into one local file, verified by full SHA256."""
import concurrent.futures
import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloud import ssh


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def copy_range(source, destination, start, count, deadline, transport=None):
    snippet = """import sys
with open(%r,'rb') as stream:
 stream.seek(%d);left=%d
 while left:
  block=stream.read(min(left,1024*1024))
  if not block:raise RuntimeError('Source shorter than admitted range')
  sys.stdout.buffer.write(block);left-=len(block)
""" % (source, start, count)
    argv = (transport or ssh()) + ['python3 -c ' + shlex.quote(snippet)]
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    fd = os.open(destination, os.O_WRONLY)
    written = 0
    try:
        while written < count:
            if time.time() >= deadline:
                raise TimeoutError('Approved backup deadline ended')
            block = process.stdout.read(min(1024 * 1024, count - written))
            if not block:
                raise RuntimeError('Incomplete remote byte range')
            offset = 0
            while offset < len(block):
                n = os.pwrite(fd, block[offset:], start + written + offset)
                if n <= 0:
                    raise RuntimeError('Local range write failed')
                offset += n
            written += len(block)
        process.wait(timeout=max(1, deadline - time.time()))
        if process.returncode or process.stdout.read(1):
            raise RuntimeError('Remote range failed or sent extra bytes')
        return {'start': start, 'bytes': written}
    finally:
        os.close(fd)
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        process.stdout.close()
        process.stderr.close()


def transfer(source, destination, size, expected_sha, deadline, workers=8, transport=None,
             chunk_size=256 * 1024 * 1024):
    destination = Path(destination)
    if destination.exists():
        if destination.stat().st_size == size and file_sha(destination) == expected_sha:
            return {'verified': True, 'already_verified': True, 'bytes': size}
        raise RuntimeError('Existing final backup differs; preserve it for diagnosis')
    partial = destination.with_name(destination.name + '.parallel-partial')
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(partial, os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.ftruncate(fd, size)
    finally:
        os.close(fd)
    chunk = chunk_size
    spans = [(start, min(chunk, size - start)) for start in range(0, size, chunk)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda span: copy_range(source, partial, *span, deadline, transport), spans))
    if sum(row['bytes'] for row in results) != size or file_sha(partial) != expected_sha:
        raise RuntimeError('Full original file SHA256 differs after range assembly')
    os.replace(partial, destination)
    return {'verified': True, 'bytes': size, 'sha256': expected_sha,
            'ranges': len(spans), 'workers': workers, 'all_original_bytes_hashed': True}
