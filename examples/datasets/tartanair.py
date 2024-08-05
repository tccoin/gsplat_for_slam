import os
from typing import Any, Dict, List, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import pathlib
import open3d as o3d
from spatialmath.base import q2r
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
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every

        # load dataset
        data_folder = pathlib.Path(data_dir)
        rgb_folder = data_folder / 'rgb'
        depth_folder = data_folder / 'depth'
        traj_file = data_folder / 'traj.tum'
        cam_file = data_folder / 'camera.yaml'
        N = sum(1 for _ in rgb_folder.iterdir()if _.is_file())
        # N=5


        # read traj file
        traj_raw = []
        with open(traj_file) as f:
            for line in f:
                traj_raw.append([float(x) for x in line.strip().split()])

        # instrinsic
        with open(cam_file) as f:
            cam_data = yaml.load(f, Loader=yaml.FullLoader)
        fx, fy, cx, cy = cam_data['fx'], cam_data['fy'], cam_data['cx'], cam_data['cy']
        image_width, image_height = cam_data['image_width'], cam_data['image_height']
        intrinsic = o3d.camera.PinholeCameraIntrinsic(image_width, image_height, fx, fy, cx, cy)
        conv_mat = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ])

        # Extract extrinsic matrices in world-to-camera format.
        w2c_mats = []
        camera_ids = []
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict() # width, height
        image_names = []
        image_paths = []
        depth_paths = []
        stacked_pc = o3d.geometry.PointCloud()
        
        # for i in range(N):
        for i in range(0,20):
            T = np.eye(4)
            T[:3, :3] = q2r(traj_raw[i][4:8], order='xyzs')
            T[:3, 3] = traj_raw[i][1:4]
            w2c_mats.append(T)
            camera_id = i
            camera_ids.append(camera_id)
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K[:2, :] /= factor
            Ks_dict[camera_id] = K
            params = np.empty(0, dtype=np.float32)
            camtype = "perspective"
            params_dict[camera_id] = params
            imsize_dict[camera_id] = (image_width // factor, image_height // factor)
            image_names.append(f'rgb_{i+1}.png')
            # load pc
            rgb_file = rgb_folder / f'rgb_{i+1}.png'
            depth_file = depth_folder / f'depth_{i+1}.png'
            image_paths.append(str(rgb_file))
            depth_paths.append(str(depth_file))

            rgb = o3d.io.read_image(str(rgb_file))
            depth = o3d.io.read_image(str(depth_file))
            depth = np.asarray(depth).astype(np.float32)
            depth = o3d.geometry.Image(depth)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth, convert_rgb_to_intensity=False, depth_trunc=100)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
            pcd = pcd.voxel_down_sample(voxel_size=0.01)
            pcd.transform(T)
            stacked_pc += pcd

        
        expected_number = 1e5
        voxel_size = 0.01
        while len(stacked_pc.points) > expected_number:
            voxel_size *= 1.1
            stacked_pc = stacked_pc.voxel_down_sample(voxel_size=voxel_size)
        points = np.asarray(stacked_pc.points, dtype=np.float32)
        points_rgb = (np.asarray(stacked_pc.colors)*255).astype(np.uint8)

        # points = np.random.rand(int(expected_number), 3)
        # points_rgb = np.random.rand(int(expected_number), 3)

        print(
            f"[Parser] {N} images."
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

        self.image_names = image_names  # List[str], (num_images,)
        self.image_paths = image_paths  # List[str], (num_images,)
        self.depth_paths = depth_paths  # List[str], (num_images,)
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
        indices = np.arange(len(self.parser.image_names))
        if split == "train":
            self.indices = indices[indices % self.parser.test_every != 0]
        else:
            self.indices = indices[indices % self.parser.test_every == 0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image = imageio.imread(self.parser.image_paths[index])[..., :3]
        depth = imageio.imread(self.parser.depth_paths[index])[..., :3]/1000.0
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
