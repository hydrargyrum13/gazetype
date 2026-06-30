# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs


project_root = Path(SPEC).resolve().parent
mediapipe_data = collect_data_files("mediapipe")
mediapipe_binaries = collect_dynamic_libs("mediapipe")
model = project_root / "src" / "gazetype" / "assets" / "face_landmarker.task"

a = Analysis(
    [str(project_root / "src" / "gazetype" / "__main__.py")],
    pathex=[str(project_root / "src")],
    binaries=mediapipe_binaries,
    datas=mediapipe_data + [(str(model), "gazetype/assets")],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter"],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Gazetype",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Gazetype",
)

