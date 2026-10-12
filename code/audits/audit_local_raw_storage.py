"""Verify raw byte provenance regardless of direct or reversible gzip storage."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time


def hash_stream(stream):
    h=hashlib.sha256()
    for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def audit(artifacts):
    manifest={};cycles=[]
    for path in sorted((artifacts/'cycle_backup_audits').glob('cycle-*.json')):
        receipt=json.loads(path.read_text())
        if receipt.get('verified'):
            manifest.update(receipt['source_manifest']);cycles.append(receipt['step'])
    full_path=artifacts/'raw_backup_audit.json'
    full=json.loads(full_path.read_text()) if full_path.exists() else None
    if full and full.get('verified'):manifest.update(full['source_manifest'])
    compressed_path=artifacts/'compressed_raw_storage.json'
    compressed=json.loads(compressed_path.read_text()) if compressed_path.exists() else {}
    mapping={r['path']:r for r in compressed.get('records',[])}
    problems=[];direct=0;stored=0;bytes_checked=0
    for name,expected in manifest.items():
        path=artifacts/'remote'/name
        if path.is_file() and not path.is_symlink():
            with path.open('rb') as stream:actual=hash_stream(stream)
            direct+=1;bytes_checked+=path.stat().st_size
        elif name in mapping:
            record=mapping[name];path=Path(record['compressed_path'])
            if record['sha256']!=expected or not path.is_file():
                problems.append({'path':name,'reason':'Compressed record missing or original identity differs'});continue
            with gzip.open(path,'rb') as stream:actual=hash_stream(stream)
            stored+=1;bytes_checked+=record['uncompressed_bytes']
        else:
            problems.append({'path':name,'reason':'Original raw bytes unavailable in direct or compressed storage'});continue
        if actual!=expected:problems.append({'path':name,'reason':'Original raw byte hash differs'})
    return {'at':time.time(),'scope':'all locally backed-up raw-cycle files and full-run manifest when present',
        'cycle_receipts':cycles,'full_run_manifest_present':bool(full and full.get('verified')),
        'files':len(manifest),'direct_files_checked':direct,'compressed_files_checked':stored,
        'uncompressed_bytes_checked':bytes_checked,'problems':problems,
        'verified':bool(manifest) and not problems,'read_only':True,
        'entire_run_complete_claimed':bool(full and full.get('verified')) and bool(manifest) and not problems}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('artifacts',type=Path)
    args=parser.parse_args();result=audit(args.artifacts)
    (args.artifacts/'local_raw_storage_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
