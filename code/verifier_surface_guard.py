"""Safety adapter for zero-corner polygons before Blender tessellation.

The published files remain unchanged. The adapter raises their own InvalidSurface
exception for structurally malformed polygon loops, preserving the existing
invalid-submission scoring/finalization path. It never changes candidate data.
"""
import inspect
import runpy
import sys
from pathlib import Path


def install(engine):
    core = Path(engine).resolve().parents[1]
    sys.path.insert(0, str(core))
    import modeling_geometry
    source = inspect.getsource(modeling_geometry.evaluated_surface)
    needle = '            mesh.calc_loop_triangles()\n'
    if source.count(needle) != 1:
        raise RuntimeError('Pinned surface function shape differs; guard not installed')
    guard = '''            # Inspect polygon topology before calling native tessellation.
            # No mesh validation/repair, deletion, or saving occurs here.
            loop_count = len(mesh.loops)
            vertex_count = len(mesh.vertices)
            for polygon in mesh.polygons:
                start, count = polygon.loop_start, polygon.loop_total
                if count < 3 or start < 0 or start + count > loop_count:
                    raise InvalidSurface("malformed polygon loop topology")
            for loop in mesh.loops:
                if loop.vertex_index < 0 or loop.vertex_index >= vertex_count:
                    raise InvalidSurface("malformed polygon vertex reference")
'''
    namespace = dict(modeling_geometry.__dict__)
    exec(compile(source.replace(needle, guard + needle), str(core / 'modeling_geometry.py'), 'exec'), namespace)
    modeling_geometry.evaluated_surface = namespace['evaluated_surface']
    return modeling_geometry


if __name__ == '__main__':
    engine = sys.argv[sys.argv.index('--') + 1]
    del sys.argv[sys.argv.index('--') + 1]
    install(engine)
    runpy.run_path(engine, run_name='__main__')
