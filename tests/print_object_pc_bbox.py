from pathlib import Path
import re

import numpy as np
import trimesh as tm


OBJECT_PC_DIR = Path(
    "/data/dataset/AnyScaleGrasp/object/DGN_2k/vision_data/azure_kinect_dk/ddg_gd_tetra_pak_poisson_008/tabletop_ur10e"
)
OBJECT_POSE = "pose000_0"


def extract_scale(dirname: str) -> int:
    match = re.match(r"scale(\d+)_", dirname)
    if match is None:
        raise ValueError(f"Could not parse scale from directory name: {dirname}")
    return int(match.group(1))


def main() -> None:
    if not OBJECT_PC_DIR.exists():
        raise FileNotFoundError(f"Directory not found: {OBJECT_PC_DIR}")

    scale_to_pointcloud = {}
    scale_to_pointcloud_file = {}
    for sample_dir in sorted(p for p in OBJECT_PC_DIR.iterdir() if p.is_dir()):
        if OBJECT_POSE not in sample_dir.name:
            continue
        scale = extract_scale(sample_dir.name)
        if scale in scale_to_pointcloud:
            continue
        pc_files = sorted(
            pc_file for pc_file in sample_dir.glob("partial_pc_*.npy") if "tabletop_ur10e" in str(pc_file)
        )
        if not pc_files:
            continue
        pc = np.load(pc_files[0], allow_pickle=True)
        pc = np.asarray(pc)
        if pc.ndim != 2 or pc.shape[1] < 3:
            raise ValueError(f"Unexpected point cloud shape {pc.shape} in {pc_files[0]}")
        scale_to_pointcloud[scale] = pc[:, :3]
        scale_to_pointcloud_file[scale] = pc_files[0]

    if not scale_to_pointcloud:
        raise RuntimeError(f"No partial point clouds found under {OBJECT_PC_DIR}")

    print(f"Object point cloud directory: {OBJECT_PC_DIR}")
    print(f"Object pose filter: {OBJECT_POSE}")
    scene_geometry = [tm.creation.axis(origin_size=0.01, axis_length=0.1)]
    x_offset = 0.0
    for scale in sorted(scale_to_pointcloud):
        points = scale_to_pointcloud[scale]
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)
        bbox_size = bbox_max - bbox_min
        print(
            f"scale{scale:03d}: "
            f"bbox_size={bbox_size.tolist()} "
            f"num_points={points.shape[0]} "
            f"source={scale_to_pointcloud_file[scale]}"
        )

        shifted_points = points.copy()
        shifted_points[:, 0] = shifted_points[:, 0] - shifted_points[:, 0].mean() + x_offset
        color = np.array(
            [
                (37 * scale) % 255,
                (91 * scale) % 255,
                (173 * scale) % 255,
                255,
            ],
            dtype=np.uint8,
        )
        colors = np.tile(color[None, :], (shifted_points.shape[0], 1))
        scene_geometry.append(tm.points.PointCloud(vertices=shifted_points, colors=colors))
        x_offset += float(bbox_size[0] + 0.08)

    tm.Scene(scene_geometry).show()


if __name__ == "__main__":
    main()
