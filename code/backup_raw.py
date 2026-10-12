"""Copy complete candidate/trajectory evidence and audit all backup file hashes."""
import shlex,subprocess
from common import *
from cloud import ssh,REMOTE

def main():
    remote_root=REMOTE+'/run'
    code="""from pathlib import Path
import hashlib,json
root=Path(%s)
result={}
for folder in ['generations','raw_turns','episodes','evaluation','jobs','train']:
 for p in (root/folder).rglob('*'):
  if p.is_file() and p.suffix!='.safetensors' and p.name!='resume.pt.gz' and not any(part.startswith(('.verified-resume-','.audit-state-reader-','.recovery-reader-')) for part in p.parts):
   h=hashlib.sha256()
   with p.open('rb') as f:
    for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
   result[str(p.relative_to(root))]=h.hexdigest()
print(json.dumps(result))
"""%repr(remote_root)
    result=subprocess.run(ssh()+['python3 -c '+shlex.quote(code)],capture_output=True,text=True,check=True,timeout=600)
    manifest=json.loads(result.stdout);target=RUN/'remote';target.mkdir(exist_ok=True)
    listfile=RUN/'raw-backup-file-list.txt';listfile.write_text('\n'.join(sorted(manifest))+'\n')
    subprocess.run(['rsync','-a','--files-from='+str(listfile),'--timeout=60','-e',shlex.join(ssh()[:-1]),
        ssh()[-1]+':'+remote_root+'/',str(target)+'/'],check=True,timeout=3600)
    failed=[name for name,expected in manifest.items() if not (target/name).exists() or sha(target/name)!=expected]
    save(RUN/'raw_backup_audit.json',{'at':time.time(),'verified':not failed,'files':len(manifest),
        'missing_or_mismatched':failed,'source_manifest':manifest})
    if failed:raise RuntimeError('Raw evidence backup failed')
if __name__=='__main__':main()
