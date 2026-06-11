# file: scripts/build_imagenet1k_kaggle_numeric.py

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

import kagglehub


DATASETS = {
    "train_0_499_a": "sautkin/imagenet1k0",
    "train_500_999_a": "sautkin/imagenet1k1",
    "train_0_499_b": "sautkin/imagenet1k2",
    "train_500_999_b": "sautkin/imagenet1k3",
    "val": "sautkin/imagenet1kvalid",
}

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".bmp", ".webp"}


def download_dataset(dataset_ref: str) -> Path:
    return Path(kagglehub.dataset_download(dataset_ref))


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def list_numeric_class_dirs(root: Path) -> list[Path]:
    class_dirs = [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
    return sorted(class_dirs, key=lambda p: int(p.name))


def copy_images(src_dir: Path, dst_dir: Path, prefix: str | None = None) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0

    for image_path in sorted(src_dir.rglob("*")):
        if not is_image(image_path):
            continue

        filename = image_path.name if prefix is None else f"{prefix}_{image_path.name}"
        dst_path = dst_dir / filename

        if dst_path.exists():
            stem = dst_path.stem
            suffix = dst_path.suffix
            index = 1
            while True:
                candidate = dst_dir / f"{stem}_{index}{suffix}"
                if not candidate.exists():
                    dst_path = candidate
                    break
                index += 1

        shutil.copy2(image_path, dst_path)
        copied += 1

    return copied


def merge_sources_to_split(
    output_root: Path,
    split_name: str,
    sources: list[tuple[str, Path]],
) -> dict[str, int]:
    split_root = output_root / split_name
    split_root.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = defaultdict(int)

    for source_tag, source_root in sources:
        class_dirs = list_numeric_class_dirs(source_root)
        print(f"[INFO] {source_tag}: {len(class_dirs)} classes trouvées dans {source_root}")

        for class_dir in class_dirs:
            class_name = class_dir.name
            target_class_dir = split_root / class_name
            copied = copy_images(
                src_dir=class_dir,
                dst_dir=target_class_dir,
                prefix=source_tag if split_name == "train" else None,
            )
            counts[class_name] += copied

    return dict(sorted(counts.items(), key=lambda kv: int(kv[0])))


def write_manifest(
    output_root: Path,
    downloads: dict[str, str],
    train_counts: dict[str, int],
    val_counts: dict[str, int],
) -> None:
    manifest = {
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
    output_root = Path("imagenet_kaggle")
    output_root.mkdir(parents=True, exist_ok=True)

    print("[INFO] Téléchargement / réutilisation du cache Kaggle...")
    downloaded = {name: download_dataset(ref) for name, ref in DATASETS.items()}
    for name, path in downloaded.items():
        print(f"[INFO] {name}: {path}")

    train_sources = [
        ("part0", downloaded["train_0_499_a"]),
        ("part1", downloaded["train_500_999_a"]),
        ("part2", downloaded["train_0_499_b"]),
        ("part3", downloaded["train_500_999_b"]),
    ]
    val_sources = [
        ("val", downloaded["val"]),
    ]

    print("\n[INFO] Construction du train/")
    train_counts = merge_sources_to_split(output_root, "train", train_sources)

    print("\n[INFO] Construction du val/")
    val_counts = merge_sources_to_split(output_root, "val", val_sources)

    write_manifest(
        output_root=output_root,
        downloads={k: str(v) for k, v in downloaded.items()},
        train_counts=train_counts,
        val_counts=val_counts,
    )

    print("\n=== Résumé ===")
    print(f"Train classes: {len(train_counts)}")
    print(f"Train images : {sum(train_counts.values())}")
    print(f"Val classes  : {len(val_counts)}")
    print(f"Val images   : {sum(val_counts.values())}")
    print(f"Output       : {output_root.resolve()}")


if __name__ == "__main__":
    main()