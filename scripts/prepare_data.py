import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def collect_images(root: Path):
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, default="data/raw")
    parser.add_argument("--out", type=str, default="data")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw = Path(args.raw)
    out = Path(args.out)

    if not raw.exists():
        raise FileNotFoundError(f"No existeix: {raw}")

    images = collect_images(raw)
    if not images:
        raise RuntimeError(f"No hi ha imatges a {raw}")

    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio ha de ser menor que 1.0")

    random.seed(args.seed)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)

    splits = {
        "train": images[:n_train],
        "val": images[n_train : n_train + n_val],
        "test": images[n_train + n_val :],
    }

    for split_name in splits:
        ensure_dir(out / split_name)

    for split_name, files in splits.items():
        dst_dir = out / split_name
        for i, src in enumerate(files):
            dst = dst_dir / f"{i:06d}_{src.stem}{src.suffix.lower()}"
            shutil.copy2(src, dst)

    print(f"Fet. Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")


if __name__ == "__main__":
    main()
