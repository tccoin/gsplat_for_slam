import os
from typing import Any, Dict, List, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import pathlib
import open3d as o3d
from spatialmath.base import q2r
from spatialmath import SE3, SO3
from .dataloader import TartanAirLoader, TUMLoader
import yaml

from .normalize import (
    align_principle_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)


def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


class Parser:
    """RGBD parser."""

    def __init__(
        self,
        data_dir: str,
        dataset_type: str = "tum",
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
    ):
        self.data_dir = data_dir
        self.dataset_type = dataset_type
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every

        # load dataset
        if dataset_type=='tartanair':
            dataset = TartanAirLoader(data_dir)
        elif dataset_type=='tum':
            dataset = TUMLoader(data_dir)

        self.dataset = dataset
        total_frames = dataset.get_total_number()
        dataset.load_ground_truth()

        # instrinsic
        fx, fy, cx, cy = dataset.camera
        image_width, image_height = dataset.image_size
        intrinsic = o3d.camera.PinholeCameraIntrinsic(image_width, image_height, fx, fy, cx, cy)

        # Extract extrinsic matrices in world-to-camera format.
        w2c_mats = []
        camera_ids = []
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict() # width, height
        stacked_pc = o3d.geometry.PointCloud()

        conv_T = np.array([
            [0,0,1,0],
            [0,1,0,0],
            [-1,0,0,0],
            [0,0,0,1]
        ])
        
        N=0
        frame_step = total_frames // 300
        print('frame_step', frame_step)
        for i in range(0, total_frames, frame_step):
            N+=1
            dataset.set_curr_index(i)
            T = dataset.read_current_ground_truth()
            T = np.array(T)
            w2c_mats.append(np.linalg.inv(T))
            camera_id = i
            camera_ids.append(camera_id)
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            K[:2, :] /= factor
            Ks_dict[camera_id] = K
            params = np.empty(0, dtype=np.float32)
            camtype = "perspective"
            params_dict[camera_id] = params
            imsize_dict[camera_id] = (image_width // factor, image_height // factor)
            # load pc
            rgb, depth = dataset.read_current_rgbd()
            rgb = o3d.geometry.Image(rgb)
            depth = o3d.geometry.Image(depth.astype(np.float32))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth, convert_rgb_to_intensity=False, depth_trunc=100, depth_scale=1)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
            pcd = pcd.voxel_down_sample(voxel_size=0.05)
            pcd.transform(T)
            stacked_pc += pcd
            print(f"point cloud {i} has {len(pcd.points)} points")

        self.total_number = N
        
        expected_number = 300_000
        voxel_size = 0.02
        while len(stacked_pc.points) > expected_number:
            voxel_size *= 1.1
            stacked_pc = stacked_pc.voxel_down_sample(voxel_size=voxel_size)
        points = np.asarray(stacked_pc.points, dtype=np.float32)
        points_rgb = (np.asarray(stacked_pc.colors)*255).astype(np.uint8)

        # T0 = np.linalg.inv(w2c_mats[0])
        # points = np.random.rand(int(expected_number), 3) * 100
        # points = np.hstack([points, np.ones((expected_number, 1))])
        # points = (T0 @ points.T).T
        # points = points[:, :3]
        # points_rgb = np.random.rand(int(expected_number), 3)*255

        print(
            f"[Parser] {N} images, {len(points)} points, voxel size {voxel_size}."
        )

        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        camtoworlds = np.linalg.inv(w2c_mats)

        # Normalize the world space.
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points = transform_points(T1, points)

            T2 = align_principle_axes(points)
            camtoworlds = transform_cameras(T2, camtoworlds)
            points = transform_points(T2, points)

            transform = T2 @ T1
        else:
            transform = np.eye(4)

        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K
        self.params_dict = params_dict  # Dict of camera_id -> params
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.points = points  # np.ndarray, (num_points, 3)
        self.points_rgb = points_rgb  # np.ndarray, (num_points, 3)
        self.transform = transform  # np.ndarray, (4, 4)

        # undistortion
        self.mapx_dict = dict()
        self.mapy_dict = dict()
        self.roi_undist_dict = dict()
        for camera_id in self.params_dict.keys():
            params = self.params_dict[camera_id]
            if len(params) == 0:
                continue  # no distortion
            assert camera_id in self.Ks_dict, f"Missing K for camera {camera_id}"
            assert (
                camera_id in self.params_dict
            ), f"Missing params for camera {camera_id}"
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]
            K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                K, params, (width, height), 0
            )
            mapx, mapy = cv2.initUndistortRectifyMap(
                K, params, None, K_undist, (width, height), cv2.CV_32FC1
            )
            self.Ks_dict[camera_id] = K_undist
            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.roi_undist_dict[camera_id] = roi_undist

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)


class Dataset:
    """A simple dataset class."""

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        indices = np.arange(len(parser.camera_ids))
        if split == "train":
            self.indices = indices[indices % self.parser.test_every != 0]
        else:
            self.indices = indices[indices % self.parser.test_every == 0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        dataset = self.parser.dataset
        index = self.indices[item]
        dataset.set_curr_index(self.parser.camera_ids[index])
        image, depth = dataset.read_current_rgbd()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
        params = self.parser.params_dict[camera_id]
        camtoworlds = self.parser.camtoworlds[index]
        factor = self.parser.factor

        if len(params) > 0:
            # Images are distorted. Undistort them.
            mapx, mapy = (
                self.parser.mapx_dict[camera_id],
                self.parser.mapy_dict[camera_id],
            )
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = self.parser.roi_undist_dict[camera_id]
            image = image[y : y + h, x : x + w]

        if factor < 1:
            # Downsample images.
            image = cv2.resize(image, (0, 0), fx=factor, fy=factor)

        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,  # the index of the image in the dataset
        }

        if self.load_depths:
            data["depths"] = torch.from_numpy(depth).float()

        return data

class MultipleDataset:
    """A dataset class that takes multiple parsers."""

    def __init__(
        self,
        parsers: list[Parser],
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
    ):
        self.parsers = parsers
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths

        seq_counts = []
        for parser in parsers:
            seq_counts.append(len(parser.camera_ids))
        seq_counts = np.array(seq_counts)
        self.seq_counts = seq_counts
        self.seq_counts_cumsum = np.cumsum(seq_counts)

        indices = np.arange(np.sum(seq_counts))

        if split == "train":
            self.indices = indices[indices % self.parser.test_every != 0]
        else:
            self.indices = indices[indices % self.parser.test_every == 0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, global_id: int) -> Dict[str, Any]:
        seq_i = np.searchsorted(self.seq_counts_cumsum, global_id, side='right')
        parser = self.parsers[seq_i]
        if seq_i == 0:
            item = global_id
        elif seq_i>0:
            item = global_id - self.seq_counts_cumsum[seq_i-1]
        dataset = self.parser.dataset
        index = self.indices[item]
        dataset.set_curr_index(self.parser.camera_ids[index])
        image, depth = dataset.read_current_rgbd()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
        params = self.parser.params_dict[camera_id]
        camtoworlds = self.parser.camtoworlds[index]
        factor = self.parser.factor

        if len(params) > 0:
            # Images are distorted. Undistort them.
            mapx, mapy = (
                self.parser.mapx_dict[camera_id],
                self.parser.mapy_dict[camera_id],
            )
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = self.parser.roi_undist_dict[camera_id]
            image = image[y : y + h, x : x + w]

        if factor < 1:
            # Downsample images.
            image = cv2.resize(image, (0, 0), fx=factor, fy=factor)

        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,  # the index of the image in the dataset
        }

        if self.load_depths:
            data["depths"] = torch.from_numpy(depth).float()

        return data


if __name__ == "__main__":
    import argparse

    import imageio.v2 as imageio
    import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/360_v2/garden")
    parser.add_argument("--dataset_type", type=str, default="tum")
    parser.add_argument("--factor", type=int, default=4)
    args = parser.parse_args()

    # Parse RGBD data.
    parser = Parser(
        data_dir=args.data_dir, factor=args.factor, normalize=True, test_every=8
    )
    dataset = Dataset(parser, split="train", load_depths=True)
    print(f"Dataset: {len(dataset)} images.")

    writer = imageio.get_writer("results/points.mp4", fps=30)
    for data in tqdm.tqdm(dataset, desc="Plotting points"):
        image = data["image"].numpy().astype(np.uint8)
        # points = data["points"].numpy()
        # depths = data["depths"].numpy()
        # for x, y in points:
        #     cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)
        writer.append_data(image)
    writer.close()
