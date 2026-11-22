# -*- mode: python ; coding: utf-8 -*-
"""Cross-platform build recipe: pyinstaller --noconfirm build.spec"""

import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all

# ---------------- tweak these ----------------
APP_NAME  = "SpellChecker"
# each language needs its _trigram/_word/_classifier files in data/models
LANGUAGES = os.environ.get("SPELLCHECKER_LANGUAGES", "xh").split()
CONSOLE   = True       # False hides the terminal window, and the error output with it
# ---------------------------------------------

IS_MAC  = sys.platform == "darwin"
ONEFILE = not IS_MAC   # a .app is a folder, so macOS must be onedir

datas = [("src", "src"), ("setup.py", ".")]
for lang in LANGUAGES:
    for kind in ("trigram", "word", "classifier"):
        pkl = Path("data/models") / f"{lang}_{kind}.pkl"
        if not pkl.exists():
            raise SystemExit(f"build.spec: missing model {pkl}")
        datas.append((str(pkl), "data/models"))

binaries = []
hiddenimports = ["sklearn.linear_model._logistic"]
_nicegui = collect_all("nicegui")
datas += _nicegui[0]
binaries += _nicegui[1]
hiddenimports += _nicegui[2]

a = Analysis(
    ["app.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["pyarrow", "pandas", "matplotlib", "mlflow", "marimo", "jupyterlab",
              "IPython", "datasets", "notebook", "pytest", "tkinter"],
    noarchive=False,
    optimize=0,
)
# the system's WebKitGTK needs the system's copies of these
if sys.platform.startswith("linux"):
    SYSTEM_LIBS = ("libstdc++.so", "libgcc_s.so", "libz.so", "libzstd.so", "libuuid.so",
                   "libffi.so", "liblzma.so", "libbz2.so")
    a.binaries = [b for b in a.binaries if not os.path.basename(b[0]).startswith(SYSTEM_LIBS)]

pyz = PYZ(a.pure)

_console = CONSOLE and not IS_MAC   # a .app has no terminal to write to

if ONEFILE:
    exe = EXE(
        pyz, a.scripts, a.binaries, a.datas, [],
        name=APP_NAME, debug=False, strip=False, upx=True, console=_console,
        bootloader_ignore_signals=False, disable_windowed_traceback=False,
        argv_emulation=False, target_arch=None, codesign_identity=None,
        entitlements_file=None, runtime_tmpdir=None,
    )
else:
    exe = EXE(
        pyz, a.scripts, [], exclude_binaries=True,
        name=APP_NAME, debug=False, strip=False, upx=True, console=_console,
        bootloader_ignore_signals=False, disable_windowed_traceback=False,
        argv_emulation=False, target_arch=None, codesign_identity=None,
        entitlements_file=None,
    )
    coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=True, name=APP_NAME)
    app = BUNDLE(coll, name=f"{APP_NAME}.app", icon=None,
                 bundle_identifier="za.ac.sun.cs.spellchecker")
