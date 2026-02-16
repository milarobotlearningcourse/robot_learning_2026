import argparse
import os
from typing import Dict, Iterable, Tuple

import h5py
import numpy as np
import datasets


def _build_poses(demo_group: h5py.Group) -> np.ndarray:
    return np.concatenate(
        (
            demo_group["obs"]["ee_pos"],
            demo_group["obs"]["ee_ori"][:, :3],
            demo_group["obs"]["gripper_states"][:, :1],
        ),
        axis=-1,
    )


def _iter_hdf5_trajectories(
    data_dir: str,
    max_trajectories: int,
) -> Iterable[Tuple[str, str]]:
    count = 0
    for root, _, files in os.walk(data_dir):
        for file in files:
            if not (file.endswith(".hdf5") or file.endswith(".h5")):
                continue
            file_path = os.path.join(root, file)
            with h5py.File(file_path, "r") as handle:
                for demo_key in handle["data"].keys():
                    yield file_path, demo_key
                    count += 1
                    if max_trajectories and count >= max_trajectories:
                        return


def _load_demo(file_path: str, demo_key: str) -> Dict[str, object]:
    with h5py.File(file_path, "r") as handle:
        demo = handle["data"][demo_key]
        images = demo["obs"]["agentview_rgb"][()]
        actions = demo["actions"][()]
        rewards = demo["rewards"][()]
        dones = demo["dones"][()]
        poses = _build_poses(demo)

    # Store as per-timestep sequences to keep variable-length episodes.
    return {
        "img": [frame.astype(np.uint8) for frame in images],
        "action": [step.astype(np.float32).tolist() for step in actions],
        "rewards": rewards.astype(np.float32).tolist(),
        "terminated": dones.astype(np.float32).tolist(),
        "poses": [step.astype(np.float32).tolist() for step in poses],
        "source_file": os.path.basename(file_path),
        "demo_key": demo_key,
    }


def _infer_shapes(sample: Dict[str, object]) -> Tuple[Tuple[int, int, int], int, int]:
    img_shape = tuple(sample["img"][0].shape)
    action_dim = int(len(sample["action"][0]))
    pose_dim = int(len(sample["poses"][0]))
    return img_shape, action_dim, pose_dim


def _make_features(img_shape: Tuple[int, int, int], action_dim: int, pose_dim: int) -> datasets.Features:
    return datasets.Features(
        {
            "img": datasets.Sequence(datasets.Array3D(shape=img_shape, dtype="uint8")),
            "action": datasets.Sequence(
                datasets.Sequence(datasets.Value("float32"), length=action_dim)
            ),
            "rewards": datasets.Sequence(datasets.Value("float32")),
            "terminated": datasets.Sequence(datasets.Value("float32")),
            "poses": datasets.Sequence(
                datasets.Sequence(datasets.Value("float32"), length=pose_dim)
            ),
            "source_file": datasets.Value("string"),
            "demo_key": datasets.Value("string"),
        }
    )


def build_dataset(data_dir: str, max_trajectories: int) -> datasets.Dataset:
    print(f"Scanning {data_dir} for HDF5 files...")
    
    # Peek at first sample to infer shapes
    iterator = _iter_hdf5_trajectories(data_dir, max_trajectories)
    try:
        first_path, first_key = next(iterator)
    except StopIteration as exc:
        raise RuntimeError(f"No HDF5 files found under {data_dir}") from exc
    
    first_sample = _load_demo(first_path, first_key)
    img_shape, action_dim, pose_dim = _infer_shapes(first_sample)
    features = _make_features(img_shape, action_dim, pose_dim)
    
    print(f"Found first trajectory. Image shape: {img_shape}, Action dim: {action_dim}, Pose dim: {pose_dim}")
    print("Loading trajectories into memory...")

    # Load all data into dict with progress
    dataset_dict: Dict[str, list] = {
        "img": [],
        "action": [],
        "rewards": [],
        "terminated": [],
        "poses": [],
        "source_file": [],
        "demo_key": [],
    }
    
    dataset_dict["img"].append(first_sample["img"])
    dataset_dict["action"].append(first_sample["action"])
    dataset_dict["rewards"].append(first_sample["rewards"])
    dataset_dict["terminated"].append(first_sample["terminated"])
    dataset_dict["poses"].append(first_sample["poses"])
    dataset_dict["source_file"].append(first_sample["source_file"])
    dataset_dict["demo_key"].append(first_sample["demo_key"])
    
    count = 1
    for file_path, demo_key in iterator:
        sample = _load_demo(file_path, demo_key)
        dataset_dict["img"].append(sample["img"])
        dataset_dict["action"].append(sample["action"])
        dataset_dict["rewards"].append(sample["rewards"])
        dataset_dict["terminated"].append(sample["terminated"])
        dataset_dict["poses"].append(sample["poses"])
        dataset_dict["source_file"].append(sample["source_file"])
        dataset_dict["demo_key"].append(sample["demo_key"])
        count += 1
        if count % 10 == 0:
            print(f"Loaded {count} trajectories...")
    
    print(f"Loaded {count} trajectories total. Creating dataset...")
    return datasets.Dataset.from_dict(dataset_dict, features=features)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export LIBERO HDF5 dataset to Hugging Face Datasets.")
    parser.add_argument(
        "--data-dir",
        default="/network/projects/real-g-grp/libero/targets_clean/",
        help="Root directory containing LIBERO HDF5 files.",
    )
    parser.add_argument("--repo-id", required=True, help="Hugging Face dataset repo ID.")
    parser.add_argument("--private", action="store_true", help="Create a private dataset repo.")
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=0,
        help="Limit number of trajectories (0 means all).",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push to the Hugging Face Hub (requires authentication).",
    )
    parser.add_argument(
        "--save-dir",
        default="",
        help="Optional local output directory for saving the dataset.",
    )
    args = parser.parse_args()

    dataset = build_dataset(args.data_dir, args.max_trajectories)

    if args.save_dir:
        dataset.save_to_disk(args.save_dir)

    if args.push:
        dataset.push_to_hub(args.repo_id, private=args.private)


if __name__ == "__main__":
    main()
