import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hemodel import load, predict_all, write

def read_config(path):
    """config.json: {"win": {...}, "size": {...}}; tuple/int keys are stored as strings."""
    if not os.path.exists(path):
        return {}, {}
    cfg = json.load(open(path))
    win = dict(cfg.get("win", {})); size = dict(cfg.get("size", {}))
    if "mainp" in win:
        win["mainp"] = {tuple(int(x) for x in k.split(",")): float(v) for k, v in win["mainp"].items()}
    if "levels" in win:
        win["levels"] = {k: tuple(v) for k, v in win["levels"].items()}
    if "lag_prior" in size:
        size["lag_prior"] = {int(k): float(v) for k, v in size["lag_prior"].items()}
    if "fam" in size:
        size["fam"] = {k: tuple(v) for k, v in size["fam"].items()}
    return win, size

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="public/ folder of the current round")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--workers", type=int, default=min(2, os.cpu_count() or 1))
    a = ap.parse_args()
    win, size = read_config(a.config)
    q, H = load(os.path.join(a.data, a.split))
    write(a.out, predict_all(q, H, win, size, workers=a.workers))

if __name__ == "__main__":
    main()
