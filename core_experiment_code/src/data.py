import gzip
import os
import random
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    num_classes: int
    train_size: int
    val_size: int
    test_size: int


def build_transforms(image_size: int = 224, resize_size: int = 256):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_tf, eval_tf


def build_dataloaders(
    config: Dict,
    dataset_key: str,
    data_root: str,
    seed: int,
    batch_size: int,
    num_workers: int,
    download: bool,
) -> DataBundle:
    image_size = int(config.get("project", {}).get("input_size", 224))
    resize_size = int(config.get("data", {}).get("eval_transform", {}).get("resize_shorter_side", 256))
    train_tf, eval_tf = build_transforms(image_size=image_size, resize_size=resize_size)
    root = Path(data_root)

    if dataset_key == "cifar100_full":
        train_set, val_set, test_set = _build_cifar100_full(root, train_tf, eval_tf, seed, download)
        num_classes = 100
    elif dataset_key == "vtab1k_cifar100":
        train_set, val_set, test_set = _build_vtab1k_cifar100(root, train_tf, eval_tf, seed, download)
        num_classes = 100
    elif dataset_key == "cub_200_2011":
        train_set, val_set, test_set = _build_cub(root, train_tf, eval_tf, seed)
        num_classes = 200
    elif dataset_key == "oxford_flowers_102":
        train_set, val_set, test_set = _build_flowers102(root, train_tf, eval_tf, download)
        num_classes = 102
    elif dataset_key == "vtab1k_flowers102":
        train_set, val_set, test_set = _build_vtab1k_flowers102(root, train_tf, eval_tf, download)
        num_classes = 102
    elif dataset_key == "snorb_azim":
        train_set, val_set, test_set = _build_smallnorb_attribute(root, train_tf, eval_tf, seed, "azimuth")
        num_classes = 18
    elif dataset_key == "snorb_elev":
        train_set, val_set, test_set = _build_smallnorb_attribute(root, train_tf, eval_tf, seed, "elevation")
        num_classes = 9
    elif dataset_key == "debug_fake":
        train_set, val_set, test_set = _build_debug_fake(train_tf, eval_tf)
        num_classes = 100
    else:
        raise ValueError(f"Unknown dataset '{dataset_key}'")

    generator = torch.Generator()
    generator.manual_seed(seed)
    pin_memory = bool(config.get("data", {}).get("pin_memory", True))
    persistent_workers = bool(config.get("data", {}).get("persistent_workers", num_workers > 0))
    prefetch_factor = config.get("data", {}).get("prefetch_factor", 4)
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=False, **loader_kwargs)
    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_classes=num_classes,
        train_size=len(train_set),
        val_size=len(val_set),
        test_size=len(test_set),
    )


def _build_cifar100_full(root: Path, train_tf, eval_tf, seed: int, download: bool):
    train_full = datasets.CIFAR100(root=str(root), train=True, transform=train_tf, download=download)
    train_eval = datasets.CIFAR100(root=str(root), train=True, transform=eval_tf, download=download)
    test_set = datasets.CIFAR100(root=str(root), train=False, transform=eval_tf, download=download)
    train_idx, val_idx = stratified_split(train_full.targets, val_ratio=0.1, seed=seed)
    return Subset(train_full, train_idx), Subset(train_eval, val_idx), test_set


def _build_vtab1k_cifar100(root: Path, train_tf, eval_tf, seed: int, download: bool):
    train_full = datasets.CIFAR100(root=str(root), train=True, transform=train_tf, download=download)
    train_eval = datasets.CIFAR100(root=str(root), train=True, transform=eval_tf, download=download)
    test_set = datasets.CIFAR100(root=str(root), train=False, transform=eval_tf, download=download)
    train_idx, val_idx = fixed_per_class_split(train_full.targets, train_per_class=10, val_per_class=10, seed=seed)
    return Subset(train_full, train_idx), Subset(train_eval, val_idx), test_set


def _build_flowers102(root: Path, train_tf, eval_tf, download: bool):
    train_set = datasets.Flowers102(root=str(root), split="train", transform=train_tf, download=download)
    val_set = datasets.Flowers102(root=str(root), split="val", transform=eval_tf, download=download)
    test_set = datasets.Flowers102(root=str(root), split="test", transform=eval_tf, download=download)
    return train_set, val_set, test_set


def _build_vtab1k_flowers102(root: Path, train_tf, eval_tf, download: bool):
    train_set = datasets.Flowers102(root=str(root), split="train", transform=train_tf, download=download)
    val_set = datasets.Flowers102(root=str(root), split="val", transform=eval_tf, download=download)
    test_set = datasets.Flowers102(root=str(root), split="test", transform=eval_tf, download=download)
    return Subset(train_set, range(800)), Subset(val_set, range(200)), test_set


def _build_smallnorb_attribute(root: Path, train_tf, eval_tf, seed: int, target: str):
    train_full = SmallNORBAttribute(root=root, split="training", target=target, transform=train_tf)
    train_eval = SmallNORBAttribute(root=root, split="training", target=target, transform=eval_tf)
    test_set = SmallNORBAttribute(root=root, split="testing", target=target, transform=eval_tf)
    train_idx, val_idx = fixed_total_stratified_split(train_full.targets, train_size=1000, val_size=1000, seed=seed)
    return Subset(train_full, train_idx), Subset(train_eval, val_idx), test_set


def _build_debug_fake(train_tf, eval_tf):
    train_set = datasets.FakeData(size=16, image_size=(3, 224, 224), num_classes=100, transform=train_tf)
    val_set = datasets.FakeData(size=8, image_size=(3, 224, 224), num_classes=100, transform=eval_tf, random_offset=1000)
    test_set = datasets.FakeData(size=8, image_size=(3, 224, 224), num_classes=100, transform=eval_tf, random_offset=2000)
    return train_set, val_set, test_set


def _build_cub(root: Path, train_tf, eval_tf, seed: int):
    train_all = CUB200(root=root, split="train", transform=train_tf)
    train_eval = CUB200(root=root, split="train", transform=eval_tf)
    test_set = CUB200(root=root, split="test", transform=eval_tf)
    labels = [label for _, label in train_all.samples]
    train_idx, val_idx = stratified_split(labels, val_ratio=0.1, seed=seed)
    return Subset(train_all, train_idx), Subset(train_eval, val_idx), test_set


class CUB200(Dataset):
    """CUB-200-2011 reader using the official metadata files."""

    def __init__(self, root: Path, split: str, transform=None) -> None:
        self.root = self._resolve_root(root)
        self.transform = transform
        if split not in {"train", "test"}:
            raise ValueError("CUB split must be 'train' or 'test'")
        want_train = split == "train"
        images = self._read_mapping("images.txt", value_type=str)
        labels = self._read_mapping("image_class_labels.txt", value_type=int)
        split_flags = self._read_mapping("train_test_split.txt", value_type=int)
        self.samples: List[Tuple[Path, int]] = []
        for image_id, rel_path in images.items():
            is_train = bool(split_flags[image_id])
            if is_train != want_train:
                continue
            label = int(labels[image_id]) - 1
            self.samples.append((self.root / "images" / rel_path, label))
        if not self.samples:
            raise RuntimeError(f"No CUB samples found for split '{split}' under {self.root}")

    @staticmethod
    def _resolve_root(root: Path) -> Path:
        direct = root / "CUB_200_2011"
        nested = root / "CUB" / "CUB_200_2011"
        if direct.exists():
            return direct
        if nested.exists():
            return nested
        raise FileNotFoundError(
            "CUB-200-2011 not found. Expected data_root/CUB_200_2011 or data_root/CUB/CUB_200_2011."
        )

    def _read_mapping(self, file_name: str, value_type):
        path = self.root / file_name
        mapping = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                key, value = line.strip().split(maxsplit=1)
                mapping[int(key)] = value_type(value)
        return mapping

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class SmallNORBAttribute(Dataset):
    """smallNORB attribute classification for VTAB-style azimuth/elevation tasks."""

    SPLIT_PREFIXES = {
        "training": "smallnorb-5x46789x9x18x6x2x96x96-training",
        "testing": "smallnorb-5x01235x9x18x6x2x96x96-testing",
    }
    TARGET_COLUMNS = {
        "elevation": 1,
        "azimuth": 2,
    }
    UINT8_MAGIC = 507333717
    INT32_MAGIC = 507333716

    def __init__(self, root: Path, split: str, target: str, transform=None, view: int = 0) -> None:
        if split not in self.SPLIT_PREFIXES:
            raise ValueError("smallNORB split must be 'training' or 'testing'")
        if target not in self.TARGET_COLUMNS:
            raise ValueError("smallNORB target must be 'azimuth' or 'elevation'")
        if view not in {0, 1}:
            raise ValueError("smallNORB view must be 0 or 1")
        self.root = self._resolve_root(root)
        self.split = split
        self.target = target
        self.transform = transform
        prefix = self.SPLIT_PREFIXES[split]
        data = self._read_matrix(self.root / f"{prefix}-dat.mat.gz")
        info = self._read_matrix(self.root / f"{prefix}-info.mat.gz")
        self.images = data[:, view, :, :]
        raw_targets = info[:, self.TARGET_COLUMNS[target]].astype(int)
        values = sorted(int(value) for value in np.unique(raw_targets))
        self.label_values = values
        label_to_index = {value: idx for idx, value in enumerate(values)}
        self.targets = [label_to_index[int(value)] for value in raw_targets]
        if len(self.images) != len(self.targets):
            raise RuntimeError("smallNORB image and target counts do not match")

    @staticmethod
    def _resolve_root(root: Path) -> Path:
        direct = root / "smallnorb"
        if direct.exists():
            return direct
        if any(root.glob("smallnorb-5x*.mat.gz")):
            return root
        raise FileNotFoundError("smallNORB not found. Expected data_root/smallnorb/*.mat.gz.")

    @classmethod
    def _read_matrix(cls, path: Path) -> np.ndarray:
        with gzip.open(path, "rb") as handle:
            magic = struct.unpack("<i", handle.read(4))[0]
            ndim = struct.unpack("<i", handle.read(4))[0]
            shape = struct.unpack("<" + "i" * ndim, handle.read(4 * ndim))
            count = int(np.prod(shape))
            if magic == cls.UINT8_MAGIC:
                dtype = np.dtype("uint8")
            elif magic == cls.INT32_MAGIC:
                dtype = np.dtype("<i4")
            else:
                raise ValueError(f"Unsupported smallNORB magic {magic} in {path}")
            data = np.frombuffer(handle.read(), dtype=dtype, count=count)
        if data.size != count:
            raise RuntimeError(f"Expected {count} values in {path}, found {data.size}")
        return data.reshape(shape).copy()

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        image = Image.fromarray(self.images[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[index]


def stratified_split(labels: Sequence[int], val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    by_class = _indices_by_class(labels)
    train_idx: List[int] = []
    val_idx: List[int] = []
    for indices in by_class.values():
        indices = list(indices)
        rng.shuffle(indices)
        val_count = max(1, int(round(len(indices) * val_ratio)))
        val_idx.extend(indices[:val_count])
        train_idx.extend(indices[val_count:])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def fixed_per_class_split(
    labels: Sequence[int],
    train_per_class: int,
    val_per_class: int,
    seed: int,
) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    for cls, indices in _indices_by_class(labels).items():
        indices = list(indices)
        rng.shuffle(indices)
        needed = train_per_class + val_per_class
        if len(indices) < needed:
            raise ValueError(f"Class {cls} has {len(indices)} samples, need {needed}")
        train_idx.extend(indices[:train_per_class])
        val_idx.extend(indices[train_per_class:needed])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def fixed_total_stratified_split(
    labels: Sequence[int],
    train_size: int,
    val_size: int,
    seed: int,
) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    by_class = _indices_by_class(labels)
    train_counts = _balanced_counts(sorted(by_class), train_size)
    val_counts = _balanced_counts(sorted(by_class), val_size)
    train_idx: List[int] = []
    val_idx: List[int] = []
    for cls in sorted(by_class):
        indices = list(by_class[cls])
        rng.shuffle(indices)
        needed = train_counts[cls] + val_counts[cls]
        if len(indices) < needed:
            raise ValueError(f"Class {cls} has {len(indices)} samples, need {needed}")
        train_idx.extend(indices[: train_counts[cls]])
        val_idx.extend(indices[train_counts[cls] : needed])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def _balanced_counts(classes: Sequence[int], total: int) -> Dict[int, int]:
    base = total // len(classes)
    remainder = total % len(classes)
    return {cls: base + (idx < remainder) for idx, cls in enumerate(classes)}


def _indices_by_class(labels: Sequence[int]) -> Dict[int, List[int]]:
    by_class: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        by_class.setdefault(int(label), []).append(idx)
    return by_class


def seed_everything(seed: int, deterministic: bool = False, benchmark: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = deterministic
