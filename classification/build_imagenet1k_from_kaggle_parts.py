from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator

import kagglehub


DATASETS: dict[str, str] = {
    "train_part_0": "sautkin/imagenet1k0",
    "train_part_1": "sautkin/imagenet1k1",
    "train_part_2": "sautkin/imagenet1k2",
    "train_part_3": "sautkin/imagenet1k3",
    "val": "sautkin/imagenet1kvalid",
}

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".bmp", ".webp"}


def is_class_dir(path: Path) -> bool:
    return path.is_dir() and path.name.startswith("n") and len(path.name) >= 9


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def find_class_dirs(root: Path) -> list[Path]:
    class_dirs: list[Path] = []

    if is_class_dir(root):
        return [root]

    for path in root.rglob("*"):
        if is_class_dir(path):
            class_dirs.append(path)

    unique = sorted({path.resolve() for path in class_dirs})
    return [Path(p) for p in unique]


def iter_images_in_class_dir(class_dir: Path) -> Iterator[Path]:
    for path in sorted(class_dir.rglob("*")):
        if is_image_file(path):
            yield path


def safe_copy(src: Path, dst_dir: Path, prefix: str | None = None) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)

    base_name = src.name
    if prefix:
        base_name = f"{prefix}_{base_name}"

    candidate = dst_dir / base_name
    if not candidate.exists():
        shutil.copy2(src, candidate)
        return candidate

    stem = Path(base_name).stem
    suffix = Path(base_name).suffix
    index = 1
    while True:
        candidate = dst_dir / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            shutil.copy2(src, candidate)
            return candidate
        index += 1


def download_dataset(dataset_ref: str) -> Path:
    path = kagglehub.dataset_download(dataset_ref)
    return Path(path)


def collect_split(
    split_name: str,
    source_roots: Iterable[Path],
    output_root: Path,
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    split_root = output_root / split_name
    split_root.mkdir(parents=True, exist_ok=True)

    for source_index, source_root in enumerate(source_roots):
        class_dirs = find_class_dirs(source_root)
        if not class_dirs:
            print(f"[WARN] Aucune classe détectée dans: {source_root}")
            continue

        print(f"[INFO] {split_name}: {len(class_dirs)} classes trouvées dans {source_root}")

        for class_dir in class_dirs:
            wnid = class_dir.name
            target_class_dir = split_root / wnid

            image_count_before = counts[wnid]
            copied_here = 0

            for image_path in iter_images_in_class_dir(class_dir):
                safe_copy(
                    src=image_path,
                    dst_dir=target_class_dir,
                    prefix=f"p{source_index}" if split_name == "train" else None,
                )
                counts[wnid] += 1
                copied_here += 1

            if copied_here == 0:
                print(f"[WARN] Pas d'images trouvées dans {class_dir}")
            elif image_count_before == 0:
                print(f"[INFO] {split_name}/{wnid}: {copied_here} images")
            else:
                print(
                    f"[INFO] {split_name}/{wnid}: +{copied_here} images "
                    f"(total {counts[wnid]})"
                )

    return dict(sorted(counts.items()))


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

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def print_summary(output_root: Path, train_counts: dict[str, int], val_counts: dict[str, int]) -> None:
    print("\n=== Résumé ===")
    print(f"Output root : {output_root.resolve()}")
    print(f"Train classes: {len(train_counts)}")
    print(f"Train images : {sum(train_counts.values())}")
    print(f"Val classes  : {len(val_counts)}")
    print(f"Val images   : {sum(val_counts.values())}")

    missing_in_val = sorted(set(train_counts) - set(val_counts))
    missing_in_train = sorted(set(val_counts) - set(train_counts))

    if missing_in_val:
        print(f"[WARN] Classes absentes de val: {len(missing_in_val)}")
    if missing_in_train:
        print(f"[WARN] Classes absentes de train: {len(missing_in_train)}")

    print("\nStructure attendue :")
    print(output_root / "train")
    print(output_root / "val")
    print(output_root / "manifest.json")


def main() -> None:
    output_root = Path("imagenet_kaggle")
    output_root.mkdir(parents=True, exist_ok=True)

    print("[INFO] Téléchargement des datasets Kaggle...")
    downloaded_paths: dict[str, Path] = {}
    for key, dataset_ref in DATASETS.items():
        path = download_dataset(dataset_ref)
        downloaded_paths[key] = path
        print(f"[INFO] {dataset_ref} -> {path}")

    train_sources = [
        downloaded_paths["train_part_0"],
        downloaded_paths["train_part_1"],
        downloaded_paths["train_part_2"],
        downloaded_paths["train_part_3"],
    ]
    val_sources = [downloaded_paths["val"]]

    print("\n[INFO] Reconstruction du split train...")
    train_counts = collect_split(
        split_name="train",
        source_roots=train_sources,
        output_root=output_root,
    )

    print("\n[INFO] Reconstruction du split val...")
    val_counts = collect_split(
        split_name="val",
        source_roots=val_sources,
        output_root=output_root,
    )

    write_manifest(
        output_root=output_root,
        downloads={k: str(v) for k, v in downloaded_paths.items()},
        train_counts=train_counts,
        val_counts=val_counts,
    )
    print_summary(output_root, train_counts, val_counts)


if __name__ == "__main__":
    main()