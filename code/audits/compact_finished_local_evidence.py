"""Reversible local gzip storage after run completion; preserve all raw bytes."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time


def hash_stream(stream):
    result=hashlib.sha256()
    for block in iter(lambda:stream.read(8*1024*1024),b''):result.update(block)
    return result.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('artifacts',type=Path)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args();root=args.artifacts/'remote'
    if args.apply and not (root/'worker_complete.json').exists():
        raise RuntimeError('Compaction requires completed sampling/training/evaluation')
    manifest={}
    for receipt in (args.artifacts/'cycle_backup_audits').glob('cycle-*.json'):
        row=json.loads(receipt.read_text())
        if row.get('verified'):manifest.update(row['source_manifest'])
    for filename in ('roft-evaluation-backup-audit.json', 'base-evaluation-snapshot-backup-audit.json'):
        snapshot=args.artifacts/filename
        if snapshot.exists():
            row=json.loads(snapshot.read_text())
            if row.get('verified'):manifest.update(row['source_manifest'])
    full=args.artifacts/'raw_backup_audit.json'
    if full.exists():
        row=json.loads(full.read_text())
        if row.get('verified'):manifest.update(row['source_manifest'])
    candidates=[]
    for name,expected in manifest.items():
        path=root/name
        # Complete original records remain available through a separately
        # qualified byte/hash reader after the live experiment has ended.
        if not path.exists() or path.is_symlink() or path.stat().st_size<64*1024:
            continue
        if (name.startswith(('generations/','raw_turns/')) and path.suffix=='.json' or
                name.startswith('episodes/') and (path.name in ['response.json','result.json'] or
                path.name.startswith('tool-') and path.suffix in ['.stdout','.stderr'] or
                '/app/' in name)):
            candidates.append((name,path,expected))
    output=args.artifacts/'compressed_raw';records=[]
    receipt=args.artifacts/'compressed_raw_storage.json'
    if args.apply and receipt.exists():
        records=json.loads(receipt.read_text()).get('records',[])
        for record in records:
            with gzip.open(record['compressed_path'],'rb') as source:
                if hash_stream(source)!=record['sha256']:
                    raise RuntimeError('Existing compressed original differs: '+record['path'])
    prior={record['path']:record for record in records}
    pending_unlinks=[]

    def commit_storage_records():
        if not pending_unlinks:
            return
        # Keep every source until its verified recovery record is published.
        # Bounded groups avoid repeatedly rewriting a growing full manifest
        # for each of the many small raw-turn files.
        temporary_receipt=receipt.with_name(receipt.name+'.tmp')
        temporary_receipt.write_text(json.dumps({'at':time.time(),'records':records,'all_raw_bytes_retained':True},indent=2)+'\n')
        temporary_receipt.replace(receipt)
        for original in pending_unlinks:
            original.unlink()
        pending_unlinks.clear()

    for name,path,expected in candidates:
        if not args.apply:
            records.append({'path':name,'uncompressed_bytes':path.stat().st_size});continue
        with path.open('rb') as stream:
            if hash_stream(stream)!=expected:raise RuntimeError('Original evidence hash changed: '+name)
        if name in prior:
            if prior[name]['sha256']!=expected:
                raise RuntimeError('Existing compressed identity differs: '+name)
            path.unlink()
            continue
        target=output/(name+'.gz');target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_name(target.name+'.tmp')
        with path.open('rb') as source,temporary.open('xb') as destination:
            with gzip.GzipFile(fileobj=destination,mode='wb',compresslevel=1,mtime=0) as compressed:
                shutil.copyfileobj(source,compressed,8*1024*1024)
        with gzip.open(temporary,'rb') as source:
            if hash_stream(source)!=expected:raise RuntimeError('Lossless round-trip differs: '+name)
        size=path.stat().st_size;temporary.replace(target)
        record={'path':name,'compressed_path':str(target),'sha256':expected,
            'uncompressed_bytes':size,'compressed_bytes':target.stat().st_size,
            'roundtrip_verified':True,'remote_original_unchanged':True}
        records.append(record)
        pending_unlinks.append(path)
        if len(pending_unlinks)>=64:
            commit_storage_records()
    if args.apply:
        commit_storage_records()
    result={'at':time.time(),'applied':args.apply,'records':records,
        'candidate_bytes':sum(r['uncompressed_bytes'] for r in records),
        'all_original_raw_bytes_retained':True,'remote_originals_removed':False}
    target=args.artifacts/('compressed_raw_storage_complete.json' if args.apply else 'compressed_raw_storage_plan.json')
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)


if __name__=='__main__':main()
