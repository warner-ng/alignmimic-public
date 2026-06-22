#!/usr/bin/env python3
"""Batch convert GLB files to OBJ via Blender.

Expected invocation (from run_hy3d_recon.py):
    blender -b -P prep/glb2obj.py -- <glb_dir> <outdir>

For each <name>.glb in <glb_dir>, exports:
    <outdir>/<name>/<name>.obj
"""

import glob
import os
import sys
import traceback

import bpy


def _clean_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _convert_one(glb_path: str, outdir: str):
    glb_basename = os.path.splitext(os.path.basename(glb_path))[0]
    target_dir = os.path.join(outdir, glb_basename)
    target_obj = os.path.join(target_dir, f"{glb_basename}.obj")

    os.makedirs(target_dir, exist_ok=True)

    _clean_scene()
    bpy.ops.import_scene.gltf(filepath=glb_path)

    # Blender 3.6+: new exporter API.
    bpy.ops.wm.obj_export(filepath=target_obj, export_materials=True)
    print(f"[OK] exported: {target_obj}")


def main():
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []

    if len(argv) != 2:
        print("usage: blender -b -P prep/glb2obj.py -- <glb_dir> <outdir>")
        return

    glb_dir, outdir = argv
    glb_dir = os.path.abspath(glb_dir)
    outdir = os.path.abspath(outdir)

    glb_files = sorted(glob.glob(os.path.join(glb_dir, "*.glb")))
    if not glb_files:
        print(f"[WARN] no .glb files found under: {glb_dir}")
        return

    failed = 0
    for glb_path in glb_files:
        try:
            _convert_one(glb_path, outdir)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[ERROR] failed converting {glb_path}: {exc}")
            traceback.print_exc()

    if failed:
        print(f"[DONE with errors] converted={len(glb_files)-failed}, failed={failed}")
    else:
        print(f"[DONE] converted all {len(glb_files)} GLB files")


if __name__ == "__main__":
    main()
