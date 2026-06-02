# file: scripts/build_flowers102_imagenet_like.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from torchvision.datasets import Flowers102


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def class_dir_name(label: int) -> str:
    return f"class_{label:03d}"


def export_split(dataset: Flowers102, split_name: str, output_root: Path) -> int:
    split_root = output_root / split_name
    ensure_dir(split_root)

    count = 0
    for index in range(len(dataset)):
        image, label = dataset[index]
        label = int(label)

        target_dir = split_root / class_dir_name(label)
        ensure_dir(target_dir)

        file_name = f"{split_name}_{index:06d}.jpg"
        output_path = target_dir / file_name

        image.save(output_path, format="JPEG")
        count += 1

    return count


def write_class_mapping(output_root: Path, labels: Iterable[int]) -> None:
    mapping = {
        str(label): class_dir_name(label)
        for label in sorted(set(int(label) for label in labels))
    }
    with (output_root / "classes.json").open("w", encoding="utf-8") as file:
        json.dump(mapping, file, indent=2, ensure_ascii=False)


def main() -> None:
    data_root = Path("data")
    output_root = Path("flowers102_imagenet_like")

    ensure_dir(data_root)
    ensure_dir(output_root)

    train_dataset = Flowers102(root=str(data_root), split="train", download=True)
    val_dataset = Flowers102(root=str(data_root), split="val", download=True)

    train_count = export_split(train_dataset, "train", output_root)
    val_count = export_split(val_dataset, "val", output_root)

    all_labels = list(train_dataset._labels) + list(val_dataset._labels)
    write_class_mapping(output_root, all_labels)

    print(f"Export terminé dans: {output_root.resolve()}")
    print(f"Train images: {train_count}")
    print(f"Val images: {val_count}")
    print("Structure créée:")
    print(output_root / "train")
    print(output_root / "val")
    print(output_root / "classes.json")


if __name__ == "__main__":
    main()