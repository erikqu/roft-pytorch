"""Trusted harness run inside the candidate container, never on the host."""
import json
from pathlib import Path
import time
import traceback

import bpy

started = time.monotonic()
result = {"status": "candidate_error", "saved": False}
destination = Path("/app/result.blend")
destination.unlink(missing_ok=True)
try:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablock in tuple(bpy.data.meshes):
        if datablock.users == 0:
            bpy.data.meshes.remove(datablock)
    code = Path("/candidate.py").read_text()
    exec(compile(code, "/candidate.py", "exec"), {"__name__": "__main__", "bpy": bpy})
    bpy.context.scene.frame_set(1)
    bpy.ops.wm.save_as_mainfile(filepath=str(destination), check_existing=False)
    result.update(status="executed_and_saved", saved=destination.is_file())
except BaseException as error:
    destination.unlink(missing_ok=True)
    result.update(error=repr(error), traceback=traceback.format_exc()[-12000:])
finally:
    result["seconds"] = time.monotonic() - started
    Path("/app/oneshot-execution.json").write_text(json.dumps(result, indent=2))
