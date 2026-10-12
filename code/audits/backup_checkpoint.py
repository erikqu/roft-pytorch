"""Off-node full checkpoint copy after numerical audit; read-only remote source."""
import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import RUN,save,load,sha
from cloud import ssh,REMOTE


def main():
    p=argparse.ArgumentParser();p.add_argument('step',type=int);a=p.parse_args()
    folder=f'checkpoint-{a.step:03d}';source=REMOTE+'/run/train/roft/'+folder
    local=RUN/'remote/train/roft'/folder
    receipt=RUN/'checkpoint_backup_audits'/(folder+'.json')
    code="from pathlib import Path;import json;p=Path("+repr(source)+");a=json.loads((p/'audit.json').read_text());print(json.dumps({'audit':a,'files':[(x.name,x.stat().st_size) for x in p.iterdir() if x.is_file()]}))"
    r=subprocess.run(ssh()+['python3 -c '+shlex.quote(code)],capture_output=True,text=True,check=True,timeout=30)
    info=json.loads(r.stdout);audit=info['audit']
    if not audit['numerical_reload_verified'] or audit['actual_changed_tensors']<=0:raise RuntimeError('Native numerical update not audited')
    required=sum(size for name,size in info['files'] if not (local/name).exists())
    if shutil.disk_usage(RUN).free<required+5*1024**3:raise RuntimeError('Host capacity inadequate; no old evidence deletion')
    local.mkdir(parents=True,exist_ok=True)
    subprocess.run(['rsync','-a','--timeout=60','-e',shlex.join(ssh()[:-1]),ssh()[-1]+':'+source+'/',str(local)+'/'],check=True,timeout=3600)
    if sha(local/'resume.pt.gz')!=audit['resume_sha256']:raise RuntimeError('FP32 optimizer/RNG backup mismatch')
    for name,expected in load(local/'weight_manifest.json').items():
        if sha(local/name)!=expected:raise RuntimeError('Full weights backup mismatch')
    save(receipt,{'at':time.time(),'verified':True,'step':a.step,'full_weights_optimizer_rng':True,
        'checkpoint_path':str(local),'resume_sha256':audit['resume_sha256'],
        'weight_manifest_sha256':sha(local/'weight_manifest.json'),'backward_tokens':audit['supervised_tokens']})
    print('CHECKPOINT_BACKUP_VERIFIED',a.step,flush=True)
if __name__=='__main__':main()
