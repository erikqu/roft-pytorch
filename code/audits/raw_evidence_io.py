"""Read original audit bytes from direct files or verified lossless local storage."""
import gzip
import hashlib
import json
from pathlib import Path


_mapping_cache = {}


def storage_records(mapping_path):
    """Index storage metadata once, invalidating on any file replacement/write.

    Original bytes are still decompressed and hashed on every read. Only the
    path index is cached; atomic compaction updates must be observed immediately.
    """
    mapping_path = Path(mapping_path)
    stamp = mapping_path.stat()
    identity = (stamp.st_dev, stamp.st_ino, stamp.st_size,
                stamp.st_mtime_ns, stamp.st_ctime_ns)
    cached = _mapping_cache.get(mapping_path)
    if cached is not None and cached[0] == identity:
        return cached[1]
    records = {}
    for row in json.loads(mapping_path.read_text()).get('records', []):
        if row['path'] in records:
            raise RuntimeError('Duplicate original-byte storage record: ' + row['path'])
        records[row['path']] = row
    _mapping_cache[mapping_path] = (identity, records)
    return records


def load_json(path):
    path = Path(path)
    if path.is_file():
        return json.loads(path.read_text())
    # Only local post-completion backups use this fallback. Native runtime
    # files are unchanged; every decompressed original is checked by hash.
    root = next((p for p in path.parents if p.name == 'remote'), None)
    if root is None:
        raise FileNotFoundError(path)
    artifacts = root.parent
    relative = str(path.relative_to(root))
    mapping_path = artifacts / 'compressed_raw_storage.json'
    if not mapping_path.is_file():
        raise FileNotFoundError(path)
    record = storage_records(mapping_path).get(relative)
    if record is None:
        raise RuntimeError('Missing or duplicate original-byte storage record: ' + relative)
    compressed = Path(record['compressed_path']).resolve()
    if not compressed.is_relative_to((artifacts / 'compressed_raw').resolve()):
        raise RuntimeError('Original evidence storage escapes backup directory')
    with gzip.open(compressed, 'rb') as stream:
        raw = stream.read()
    if len(raw) != record['uncompressed_bytes'] or hashlib.sha256(raw).hexdigest() != record['sha256']:
        raise RuntimeError('Decompressed original evidence identity differs: ' + relative)
    return json.loads(raw)


def has_json(path):
    if Path(path).is_file():
        return True
    try:
        load_json(path)
    except FileNotFoundError:
        return False
    return True


def json_paths(folder, pattern):
    """Enumerate original JSON names in direct and verified compressed storage."""
    import fnmatch
    folder = Path(folder)
    paths = set(folder.glob(pattern))
    root = next((p for p in [folder, *folder.parents] if p.name == 'remote'), None)
    if root is not None:
        mapping = root.parent / 'compressed_raw_storage.json'
        if mapping.is_file():
            for record in storage_records(mapping).values():
                original = root / record['path']
                if original.parent == folder and fnmatch.fnmatchcase(original.name, pattern):
                    paths.add(original)
    return sorted(paths)
