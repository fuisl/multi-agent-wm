"""Populate $STABLEWM_HOME with the original LeWM datasets and checkpoints.

Source: https://huggingface.co/collections/quentinll/lewm

    python scripts/download_lewm.py pusht              # dataset + checkpoint
    python scripts/download_lewm.py pusht --no-data    # checkpoint only (~70MB)
    python scripts/download_lewm.py all

Layout after download ($STABLEWM_HOME defaults to <repo>/data):
    datasets/pusht_expert_train.h5             <- dataset (tworoom / cube / reacher: extracted archives)
    checkpoints/lewm/pusht/{weights.pt,config.json}
"""

import argparse
import json
import os
import shutil
import tarfile
from pathlib import Path

os.environ.setdefault("STABLEWM_HOME", str(Path(__file__).resolve().parents[1] / "data"))

import zstandard
from huggingface_hub import hf_hub_download, list_repo_files

REPOS = {
    "pusht": "quentinll/lewm-pusht",
    "tworoom": "quentinll/lewm-tworooms",
    "cube": "quentinll/lewm-cube",
    "reacher": "quentinll/lewm-reacher",
}

# the HF configs target swm's refactored copy; point them at the official le-wm classes (jepa.py / module.py)
TARGETS = {
    "stable_worldmodel.wm.lewm.LeWM": "jepa.JEPA",
    "stable_worldmodel.wm.lewm.module.Predictor": "module.ARPredictor",
    "stable_worldmodel.wm.lewm.module.Embedder": "module.Embedder",
    "stable_worldmodel.wm.lewm.module.MLP": "module.MLP",
}


def retarget(cfg):
    if isinstance(cfg, dict):
        return {k: TARGETS.get(v, v) if k == "_target_" else retarget(v) for k, v in cfg.items()}
    return cfg


def decompress(src: Path, dst_dir: Path):
    """.h5.zst -> .h5, .tar.zst -> extracted tree (streamed, no temp copy)."""
    dctx = zstandard.ZstdDecompressor()
    with open(src, "rb") as f, dctx.stream_reader(f) as reader:
        if src.name.endswith(".tar.zst"):
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                tar.extractall(dst_dir)
        else:
            with open(dst_dir / src.name.removesuffix(".zst"), "wb") as out:
                shutil.copyfileobj(reader, out, length=64 << 20)


def download_data(repo: str, root: Path, keep_archive: bool):
    archive = next(f for f in list_repo_files(repo, repo_type="dataset") if f.endswith(".zst"))
    out = root / "datasets"
    out.mkdir(parents=True, exist_ok=True)
    marker = out / f".{archive}.done"
    if marker.exists():
        print(f"[data] {archive} already extracted")
        return
    print(f"[data] downloading {repo}/{archive}")
    path = Path(hf_hub_download(repo, archive, repo_type="dataset", local_dir=root / "archives"))
    print(f"[data] decompressing into {out}")
    decompress(path, out)
    marker.touch()
    if not keep_archive:
        path.unlink()


def download_ckpt(repo: str, env: str, root: Path):
    out = root / "checkpoints" / "lewm" / env
    out.mkdir(parents=True, exist_ok=True)
    hf_hub_download(repo, "weights.pt", local_dir=out)
    cfg = json.loads(Path(hf_hub_download(repo, "config.json", local_dir=out)).read_text())
    (out / "config.json").write_text(json.dumps(retarget(cfg), indent=2))
    print(f"[ckpt] {out}  ->  eval.py policy=lewm/{env}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("envs", nargs="+", choices=[*REPOS, "all"])
    parser.add_argument("--no-data", action="store_true", help="skip the (large) datasets")
    parser.add_argument("--no-ckpt", action="store_true", help="skip the checkpoints")
    parser.add_argument("--keep-archive", action="store_true", help="keep the .zst after extraction")
    args = parser.parse_args()

    root = Path(os.environ["STABLEWM_HOME"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    envs = list(REPOS) if "all" in args.envs else args.envs

    for env in envs:
        if not args.no_ckpt:
            download_ckpt(REPOS[env], env, root)
        if not args.no_data:
            download_data(REPOS[env], root, args.keep_archive)


if __name__ == "__main__":
    main()
