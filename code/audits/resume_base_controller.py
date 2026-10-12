"""Bounded, single-driver continuation of frozen base inference and judging."""
import fcntl
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from common import RUN, load, save, config, append
from cloud import ssh, owned, REMOTE, PYTHON, NODE_PYTHON
from controller import sync, remote


def main():
    with (RUN / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        approval = load(RUN / 'work_window_extension_approval.json')
        deadline = load(RUN / 'work_window.json')['deadline_unix']
        if approval['approved_deadline_unix'] != deadline or time.time() >= deadline:
            raise RuntimeError('Explicit approved continuation window is unavailable')
        bridge = tunnel = None
        pid = None
        try:
            bridge = subprocess.Popen([PYTHON, '-u', str(CODE / 'judge_bridge.py')],
                stdout=(RUN / 'logs/judge_bridge.log').open('ab'), stderr=subprocess.STDOUT)
            port = config()['judge_port']
            tunnel = subprocess.Popen(ssh()[:-1] + ['-o', 'ExitOnForwardFailure=yes',
                '-N', '-R', f'{port}:127.0.0.1:{port}', ssh()[-1]],
                stdout=subprocess.DEVNULL, stderr=(RUN / 'logs/tunnel.log').open('ab'))
            time.sleep(2)
            if bridge.poll() is not None or tunnel.poll() is not None:
                raise RuntimeError('Existing judge ledger transport did not start')
            env = {'ROFT_RUN': REMOTE + '/run', 'ROFT_MODEL': REMOTE + '/model',
                'TOKENIZERS_PARALLELISM': 'false',
                'LD_LIBRARY_PATH': os.environ.get('ROFT_NODE_LD_LIBRARY_PATH', ''),
                'PATH': os.environ.get('ROFT_NODE_PATH', str(Path(NODE_PYTHON).parent)+':/usr/local/cuda/bin:/usr/bin:/bin')}
            launch = "import subprocess,os;env=dict(os.environ);env.update(" + repr(env) + ");" + \
                "log=open('" + REMOTE + "/worker.log','ab');p=subprocess.Popen([" + \
                repr(NODE_PYTHON) + ",'-u'," + \
                repr(REMOTE + '/code/audits/resume_base_worker.py') + \
                "],env=env,stdout=log,stderr=log,start_new_session=True);print(p.pid)"
            pid = int(remote('python3 -c ' + shlex.quote(launch)).strip())
            save(RUN / 'base_continuation_launch.json', {'at': time.time(),
                'controller_pid': os.getpid(), 'worker_pid': pid, 'bridge_pid': bridge.pid,
                'tunnel_pid': tunnel.pid, 'approved_deadline_unix': deadline,
                'training_repeated': False, 'automatic_shutdown': False})
            while time.time() < deadline:
                try:
                    sync()
                except (subprocess.TimeoutExpired, RuntimeError, subprocess.CalledProcessError) as error:
                    append(RUN / 'base_continuation_backup_failures.jsonl', {
                        'at': time.time(), 'error': str(error)[:600],
                        'worker_stopped': False, 'will_retry': True})
                if (RUN / 'remote/worker_complete.json').exists():
                    return
                if (RUN / 'remote/failure.json').exists():
                    raise RuntimeError('Continuation failed; original evidence retained')
                if bridge.poll() is not None or tunnel.poll() is not None:
                    raise RuntimeError('Judge bridge/tunnel exited')
                time.sleep(20)
            raise TimeoutError('Approved continuation deadline ended; node retained')
        except BaseException as error:
            save(RUN / 'base_continuation_controller_failure.json', {'at': time.time(),
                'type': type(error).__name__, 'error': str(error), 'node_kept_running': True})
            raise
        finally:
            if pid and not (RUN / 'remote/worker_complete.json').exists():
                stop = "import os,signal;from pathlib import Path;p=Path('/proc/" + str(pid) + "/cmdline');" + \
                    "os.killpg(" + str(pid) + ",signal.SIGTERM) if p.exists() and " + \
                    "b'audits/resume_base_worker.py' in p.read_bytes() else None"
                try:
                    remote('python3 -c ' + shlex.quote(stop))
                except Exception:
                    pass
            try:
                sync()
            except (subprocess.TimeoutExpired, RuntimeError, subprocess.CalledProcessError) as error:
                append(RUN / 'base_continuation_backup_failures.jsonl', {
                    'at': time.time(), 'error': str(error)[:600],
                    'cleanup_sync_failure': True, 'node_kept_running': True})
            finally:
                if tunnel:
                    tunnel.terminate()
                if bridge:
                    bridge.terminate()
                node = owned()
                save(RUN / 'node_kept_running.json', {'at': time.time(),
                    'state': node['status'] if node else 'missing', 'automatic_shutdown': False,
                    'instance_id': node['id'] if node else None, 'still_billable': bool(node),
                    'reason': 'Explicit user instruction after authorized continuation'})


if __name__ == '__main__':
    main()
