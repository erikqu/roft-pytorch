"""Additional lossless off-node archive; never removes original backup files."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tarfile
import time


def digest_file(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('artifacts',type=Path)
    parser.add_argument('--through-step',type=int,required=True)
    args=parser.parse_args();root=args.artifacts/'remote';manifest={}
    for step in range(1,args.through_step+1):
        receipt=json.loads((args.artifacts/'cycle_backup_audits'/f'cycle-{step:03d}.json').read_text())
        if not receipt['verified']:raise RuntimeError('Incomplete cycle backup')
        manifest.update(receipt['source_manifest'])
    destination=args.artifacts/'raw_archives';destination.mkdir(exist_ok=True)
    target=destination/f'cycles-through-{args.through_step:03d}.tar.gz'
    if target.exists():raise RuntimeError('Archive already exists; inspect prior receipt')
    temporary=target.with_name(target.name+'.tmp')
    with temporary.open('xb') as output,gzip.GzipFile(fileobj=output,mode='wb',compresslevel=1,mtime=0) as compressed:
        with tarfile.open(fileobj=compressed,mode='w|') as archive:
            for name,expected in sorted(manifest.items()):
                path=root/name
                if path.is_symlink() or digest_file(path)!=expected:
                    raise RuntimeError('Original immutable file changed: '+name)
                archive.add(path,arcname=name,recursive=False)
    checked=set();raw_bytes=0
    with gzip.open(temporary,'rb') as compressed,tarfile.open(fileobj=compressed,mode='r|') as archive:
        for member in archive:
            if not member.isfile() or member.name in checked or member.name not in manifest:
                raise RuntimeError('Unexpected archive member: '+member.name)
            h=hashlib.sha256()
            stream=archive.extractfile(member)
            for block in iter(lambda:stream.read(8*1024*1024),b''):
                h.update(block);raw_bytes+=len(block)
            if h.hexdigest()!=manifest[member.name]:
                raise RuntimeError('Archive round-trip bytes differ: '+member.name)
            checked.add(member.name)
    if checked!=set(manifest):raise RuntimeError('Archive membership differs from full cycles')
    temporary.replace(target)
    receipt={'at':time.time(),'archive':str(target),'archive_sha256':digest_file(target),
        'through_step':args.through_step,'files':len(checked),'uncompressed_bytes':raw_bytes,
        'archive_bytes':target.stat().st_size,'source_manifest':manifest,
        'all_member_bytes_hashverified':True,'original_local_files_removed':False,
        'remote_originals_changed':False,'verified':True}
    path=destination/f'cycles-through-{args.through_step:03d}-audit.json'
    path.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='source_manifest'}),flush=True)


if __name__=='__main__':main()
