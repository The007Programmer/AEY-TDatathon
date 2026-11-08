"""Freeze the current bot as train/snapshots/<name>/agent.py (package renamed) for A/B tests.

    .venv/bin/python -m train.snapshot v2
    .venv/bin/python -m train.arena agent.py train/snapshots/v2/agent.py
"""
import os, re, shutil, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
name = sys.argv[1]
dst = os.path.join(ROOT, "train", "snapshots", name)
pkg = f"bot_{name}"
shutil.rmtree(dst, ignore_errors=True)
os.makedirs(dst)
shutil.copytree(os.path.join(ROOT, "bot"), os.path.join(dst, pkg), ignore=shutil.ignore_patterns("__pycache__"))
os.symlink(os.path.join(ROOT, "data"), os.path.join(dst, "data"))
for f in os.listdir(os.path.join(dst, pkg)):
    p = os.path.join(dst, pkg, f)
    if f.endswith(".py"):
        s = open(p).read()
        s = re.sub(r"\bfrom bot(\.| )", lambda m: f"from {pkg}{m.group(1)}", s)
        s = s.replace("from bot import", f"from {pkg} import")
        open(p, "w").write(s)
s = open(os.path.join(ROOT, "agent.py")).read()
s = s.replace("from bot import brain as _brain", f"import sys as _s, os as _o\n    _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))\n    from {pkg} import brain as _brain")
open(os.path.join(dst, "agent.py"), "w").write(s)
print("snapshot ->", dst)
