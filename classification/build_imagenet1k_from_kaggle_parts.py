# file: scripts/build_imagenet1k_kaggle_numeric_fast.py

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import kagglehub
from tqdm import tqdm


DATASETS = {
    "train_0_499_a": "sautkin/imagenet1k0",
    "train_500_999_a": "sautkin/imagenet1k1",
    "train_0_499_b": "sautkin/imagenet1k2",
    "train_500_999_b": "sautkin/imagenet1k3",
    "val": "sautkin/imagenet1kvalid",
}

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="imagenet_kaggle", type=str)
    parser.add_argument(
        "--mode",
        default="copy",
        choices=["symlink", "hardlink", "copy"],
        help="symlink = le plus rapide, hardlink = rapide si même filesystem, copy = copie réelle",
    )
    parser.add_argument(
        "--workers",
        default=min(32, (os.cpu_count() or 8)),
        type=int,
        help="nombre de workers pour création/copie des fichiers",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="supprime output-root avant reconstruction",
    )
    return parser.parse_args()


def download_dataset(dataset_ref: str) -> Path:
    return Path(kagglehub.dataset_download(dataset_ref))


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def list_numeric_class_dirs(root: Path) -> list[Path]:
    return sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )


def iter_images(root: Path) -> list[Path]:
    return [p for p in sorted(root.rglob("*")) if is_image(p)]


def ensure_empty_dir(path: Path, force: bool) -> None:
    if path.exists() and force:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def materialize_file(src: Path, dst: Path, mode: str) -> None:
    if dst.exists():
        return

    dst.parent.mkdir(parents=True, exist_ok=True)

    if mode == "symlink":
        dst.symlink_to(src.resolve())
        return

    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            shutil.copy2(src, dst)
            return

    shutil.copy2(src, dst)


def build_tasks_for_sources(
    split_name: str,
    sources: list[tuple[str, Path]],
    output_root: Path,
) -> tuple[list[tuple[Path, Path]], dict[str, int]]:
    tasks: list[tuple[Path, Path]] = []
    counts: dict[str, int] = defaultdict(int)
    split_root = output_root / split_name

    for source_tag, source_root in sources:
        class_dirs = list_numeric_class_dirs(source_root)
        print(f"[INFO] {source_tag}: {len(class_dirs)} classes trouvées dans {source_root}")

        for class_dir in class_dirs:
            class_name = class_dir.name
            images = iter_images(class_dir)

            for src in images:
                filename = src.name if split_name == "val" else f"{source_tag}_{src.name}"
                dst = split_root / class_name / filename
                tasks.append((src, dst))
                counts[class_name] += 1

    return tasks, dict(sorted(counts.items(), key=lambda kv: int(kv[0])))


def execute_tasks(
    tasks: list[tuple[Path, Path]],
    mode: str,
    workers: int,
    desc: str,
) -> None:
    def _job(task: tuple[Path, Path]) -> None:
        src, dst = task
        materialize_file(src, dst, mode)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(tqdm(executor.map(_job, tasks), total=len(tasks), desc=desc))


def write_manifest(
    output_root: Path,
    downloads: dict[str, str],
    train_counts: dict[str, int],
    val_counts: dict[str, int],
    mode: str,
) -> None:
    manifest = {
        "mode": mode,
        "downloads": downloads,
        "splits": {
            "train": {
                "num_classes": len(train_counts),
                "num_images": sum(train_counts.values()),
                "per_class_counts": train_counts,
            },
            "val": {
                "num_classes": len(val_counts),
                "num_images": sum(val_counts.values()),
                "per_class_counts": val_counts,
            },
        },
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)

    if output_root.exists() and not args.force:
        manifest = output_root / "manifest.json"
        if manifest.exists():
            print(f"[INFO] {output_root.resolve()} existe déjà.")
            print("[INFO] Utilise --force pour reconstruire.")
            return

    ensure_empty_dir(output_root, force=args.force)
    ensure_empty_dir(output_root / "train", force=False)
    ensure_empty_dir(output_root / "val", force=False)

    print("[INFO] Téléchargement / réutilisation du cache KaggleHub...")
    downloaded = {name: download_dataset(ref) for name, ref in DATASETS.items()}
    for name, path in downloaded.items():
        print(f"[INFO] {name}: {path}")

    train_sources = [
        ("part0", downloaded["train_0_499_a"]),
        ("part1", downloaded["train_500_999_a"]),
        ("part2", downloaded["train_0_499_b"]),
        ("part3", downloaded["train_500_999_b"]),
    ]
    val_sources = [("val", downloaded["val"])]

    print("\n[INFO] Indexation du split train...")
    train_tasks, train_counts = build_tasks_for_sources(
        split_name="train",
        sources=train_sources,
        output_root=output_root,
    )

    print("[INFO] Indexation du split val...")
    val_tasks, val_counts = build_tasks_for_sources(
        split_name="val",
        sources=val_sources,
        output_root=output_root,
    )

    print(f"\n[INFO] Reconstruction train ({args.mode})...")
    execute_tasks(
        tasks=train_tasks,
        mode=args.mode,
        workers=args.workers,
        desc="train",
    )

    print(f"[INFO] Reconstruction val ({args.mode})...")
    execute_tasks(
        tasks=val_tasks,
        mode=args.mode,
        workers=args.workers,
        desc="val",
    )

    write_manifest(
        output_root=output_root,
        downloads={k: str(v) for k, v in downloaded.items()},
        train_counts=train_counts,
        val_counts=val_counts,
        mode=args.mode,
    )

    print("\n=== Résumé ===")
    print(f"Mode         : {args.mode}")
    print(f"Train classes: {len(train_counts)}")
    print(f"Train images : {sum(train_counts.values())}")
    print(f"Val classes  : {len(val_counts)}")
    print(f"Val images   : {sum(val_counts.values())}")
    print(f"Output       : {output_root.resolve()}")


if __name__ == "__main__":
    main()