#!/usr/bin/python3
"""Preserve sealed command arguments while installing the qualified surface guard."""
import os
import sys

arguments = sys.argv[1:]
if '--python' in arguments:
    index = arguments.index('--python') + 1
    if arguments[index].endswith('/engines/standard_scene.py'):
        engine = arguments[index]
        arguments[index] = '/opt/roft-verifier-guard.py'
        separator = arguments.index('--') + 1
        arguments.insert(separator, engine)
os.execv('/opt/blender-4.5.12-linux-x64/blender',
         ['/opt/blender-4.5.12-linux-x64/blender', *arguments])
