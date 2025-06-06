# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
import functools
import io
from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh.transformations as tra
from PIL import Image
from torch import Tensor


def compute_pinhole_K(width, height, fov_degrees) -> np.ndarray:
    """Create a simple pinhole camera given minimal information only. Fov is in degrees.

    Args:
        width (float): width of image
        height (float): height of image
        fov_degrees (float): field of view in degrees

    Returns:
        np.ndarray: 3x3 camera intrinsics matrix
    """
    horizontal_fov_rad = np.radians(fov_degrees)
    h_focal_length = width / (2 * np.tan(horizontal_fov_rad / 2))
    v_focal_length = width / (2 * np.tan(horizontal_fov_rad / 2) * float(height) / width)
    principal_point_x = (width - 1.0) / 2
    principal_point_y = (height - 1.0) / 2
    K = np.array(
        [[v_focal_length, 0, principal_point_x], [0, h_focal_length, principal_point_y], [0, 0, 1]]
    )
    return K


class Camera(object):
    """
    Simple pinhole camera model. Contains parameters for projecting from depth to xyz, and saving information about camera position for planning.
    TODO: Move this to utils/cameras.py?
    """

    @staticmethod
    def from_width_height_fov(
        width: float,
        height: float,
        fov_degrees: float,
        near_val: float = 0.1,
        far_val: float = 4.0,
    ):
        """Create a simple pinhole camera given minimal information only. Fov is in degrees"""
        horizontal_fov_rad = np.radians(fov_degrees)
        h_focal_length = width / (2 * np.tan(horizontal_fov_rad / 2))
        v_focal_length = width / (2 * np.tan(horizontal_fov_rad / 2) * float(height) / width)
        principal_point_x = (width - 1.0) / 2
        principal_point_y = (height - 1.0) / 2
        return Camera(
            (0, 0, 0),
            (0, 0, 0, 1),
            height,
            width,
            v_focal_length,
            h_focal_length,
            principal_point_x,
            principal_point_y,
            near_val,
            far_val,
            np.eye(4),
            None,
            None,
            horizontal_fov_rad,
        )

    @staticmethod
    def from_K(K: np.ndarray, width: float, height: float):
        """return camera created from a 3x3 camera intrinsics matrix K"""
        assert K.shape == (3, 3)
        return Camera(
            (0, 0, 0),
            (0, 0, 0, 1),
            height,
            width,
            K[0, 0],
            K[1, 1],
            K[0, 2],
            K[1, 2],
            0,
            5,
            np.eye(4),
            None,
            None,
            None,
        )

    def __init__(
        self,
        pos,
        orn,
        height,
        width,
        fx,
        fy,
        px,
        py,
        near_val,
        far_val,
        pose_matrix,
        proj_matrix,
        view_matrix,
        fov,
        *args,
        **kwargs,
    ):
        self.pos = pos
        self.orn = orn
        self.height = height
        self.width = width
        self.px = px
        self.py = py
        self.fov = fov
        self.near_val = near_val
        self.far_val = far_val
        self.fx = fx
        self.fy = fy
        self.pose_matrix = pose_matrix
        self.pos = pos
        self.orn = orn
        # symmetric pinhole should have the same xy focal length
        self.K = np.array([[self.fy, 0, self.px], [0, self.fy, self.py], [0, 0, 1]])

    def to_dict(self):
        """create a dictionary so that we can extract the necessary information for
        creating point clouds later on if we so desire"""
        info = {}
        info["pos"] = self.pos
        info["orn"] = self.orn
        info["height"] = self.height
        info["width"] = self.width
        info["near_val"] = self.near_val
        info["far_val"] = self.far_val
        info["proj_matrix"] = self.proj_matrix
        info["view_matrix"] = self.view_matrix
        info["max_depth"] = self.max_depth
        info["pose_matrix"] = self.pose_matrix
        info["px"] = self.px
        info["py"] = self.py
        info["fx"] = self.fx
        info["fy"] = self.fy
        info["fov"] = self.fov
        return info

    def get_pose(self):
        return self.pose_matrix.copy()

    def depth_to_xyz(self, depth, data_type: type = np.float16):
        """get depth from numpy using simple pinhole self model"""
        indices = np.indices((self.height, self.width), dtype=np.float32).transpose(1, 2, 0)
        z = depth
        # pixel indices start at top-left corner. for these equations, it starts at bottom-left
        x = (indices[:, :, 1] - self.px) * (z / self.fx)
        y = (indices[:, :, 0] - self.py) * (z / self.fy)
        # Should now be height x width x 3, after this:
        xyz = np.stack([x, y, z], axis=-1).astype(data_type)
        return xyz

    def fix_depth(self, depth):
        if isinstance(depth, np.ndarray):
            depth = depth.copy()
        else:
            # Assuming it's a torch tensor instead
            depth = depth.clone()

        depth[depth > self.far_val] = 0
        depth[depth < self.near_val] = 0
        return depth


def camera_xyz_to_global_xyz(camera_xyz, camera_pose):
    """
    camera_xyz (height, width, 3)
    camera_pose (4, 4)
    """
    height, width, _ = camera_xyz.shape

    camera_xyz_flat = camera_xyz.reshape(-1, 3)
    ones = np.ones((camera_xyz_flat.shape[0], 1))
    camera_homogeneous = np.hstack((camera_xyz_flat, ones))  # Shape (N, 4)

    global_homogeneous = (camera_pose @ camera_homogeneous.T).T  # Shape (N, 4)

    # Convert back to Cartesian coordinates
    global_xyz_flat = global_homogeneous[:, :3] / global_homogeneous[:, 3:4]  # Shape (N, 3)

    # Reshape back to (height, width, 3)
    global_xyz = global_xyz_flat.reshape(height, width, 3)

    return global_xyz


def z_from_opengl_depth(depth, camera: Camera):
    near = camera.near_val
    far = camera.far_val
    # return (2.0 * near * far) / (near + far - depth * (far - near))
    return (near * far) / (far - depth * (far - near))


# We apply this correction to xyz when computing it in sim
# R_CORRECTION = R1 @ R2
T_CORRECTION = tra.euler_matrix(0, 0, np.pi / 2)
R_CORRECTION = T_CORRECTION[:3, :3]


def opengl_to_opencv(pose):
    transform = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    pose = pose @ transform
    return pose


def convert_xz_y_to_xyz(camera_pose_xz_y):
    # Extract rotation matrix and translation vector from the camera pose
    rotation_matrix_xz_y = camera_pose_xz_y[:3, :3]
    translation_vector_xz_y = camera_pose_xz_y[:3, 3]

    # Convert rotation matrix from XZ-Y to XYZ convention
    rotation_matrix_xyz = np.array(
        [
            [
                rotation_matrix_xz_y[0, 0],
                rotation_matrix_xz_y[0, 1],
                rotation_matrix_xz_y[0, 2],
            ],
            [
                -rotation_matrix_xz_y[2, 0],
                -rotation_matrix_xz_y[2, 1],
                -rotation_matrix_xz_y[2, 2],
            ],
            [
                rotation_matrix_xz_y[1, 0],
                rotation_matrix_xz_y[1, 1],
                rotation_matrix_xz_y[1, 2],
            ],
        ]
    )

    # Convert translation vector from XZ-Y to XYZ convention
    translation_vector_xyz = np.array(
        [
            translation_vector_xz_y[0],
            -translation_vector_xz_y[2],
            translation_vector_xz_y[1],
        ]
    )

    # Create the new camera pose matrix in XYZ convention
    camera_pose_xyz = np.eye(4)
    camera_pose_xyz[:3, :3] = rotation_matrix_xyz
    camera_pose_xyz[:3, 3] = translation_vector_xyz

    return camera_pose_xyz


def opengl_depth_to_xyz(depth, camera: Camera):
    """get depth from numpy using simple pinhole camera model"""
    indices = np.indices((camera.height, camera.width), dtype=np.float32).transpose(1, 2, 0)
    z = depth
    # pixel indices start at top-left corner. for these equations, it starts at bottom-left
    # indices[..., 0] = np.flipud(indices[..., 0])
    x = (indices[:, :, 1] - camera.px) * (z / camera.fx)
    y = (indices[:, :, 0] - camera.py) * (z / camera.fy)  # * -1
    # Should now be height x width x 3, after this:
    xyz = np.stack([x, y, z], axis=-1) @ R_CORRECTION
    return xyz


def depth_to_xyz(depth, camera: Camera):
    """get depth from numpy using simple pinhole camera model"""
    indices = np.indices((camera.height, camera.width), dtype=np.float32).transpose(1, 2, 0)
    z = depth
    # pixel indices start at top-left corner. for these equations, it starts at bottom-left
    x = (indices[:, :, 1] - camera.px) * (z / camera.fx)
    y = (indices[:, :, 0] - camera.py) * (z / camera.fy)
    # Should now be height x width x 3, after this:
    xyz = np.stack([x, y, z], axis=-1)
    return xyz


def smooth_mask(mask, kernel=None, num_iterations=3):
    """Dilate and then erode.

    Arguments:
        mask: the mask to clean up

    Returns:
        mask: the dilated mask
        mask2: dilated, then eroded mask
    """
    if kernel is None:
        kernel = np.ones((5, 5))
    mask = mask.astype(np.uint8)
    mask1 = cv2.dilate(mask, kernel, iterations=num_iterations)
    # second step
    mask2 = mask
    mask2 = cv2.erode(mask2, kernel, iterations=num_iterations)
    mask2 = np.bitwise_and(mask, mask2)
    return mask1, mask2


def rotate_image(imgs: List[np.ndarray]) -> List[np.ndarray]:
    """stretch specific routine to flip and rotate sideways images for normal viewing"""
    imgs = [np.rot90(np.fliplr(np.flipud(x))) for x in imgs]
    return imgs


def build_mask(
    target: Tensor, val: float = 0.0, tol: float = 1e-3, mask_extra_radius: int = 5
) -> Tensor:
    """Build mask where all channels are (val - tol) <= target <= (val + tol)
        Optionally, dilate by mask_extra_radius

    Args:
        target (Tensor): [B, N_channels, H, W] input tensor
        val (float): Value to use for masking. Defaults to 0.0.
        tol (float): Tolerance for mask. Defaults to 1e-3.
        mask_extra_radius (int, optional): Dilate by mask_extra_radius pix . Defaults to 5.

    Returns:
        _type_: Mask of shape target.shape
    """
    assert target.ndim == 4, f"target should be of shape [B, N_channels, H, W], was {target.shape}"
    if target.shape[1] == 1:
        masks = [target[:, t] for t in range(target.shape[1])]
        masks = [(t >= val - tol) & (t <= val + tol) for t in masks]
        mask = functools.reduce(lambda a, b: a & b, masks).unsqueeze(1)
    else:
        mask = (target >= val - tol) & (target <= val + tol)
    mask = 0 != F.conv2d(
        mask.float(),
        torch.ones(1, 1, mask_extra_radius, mask_extra_radius, device=mask.device),
        padding=(mask_extra_radius // 2),
    )  # type: ignore
    return (~mask).expand_as(target)


def dilate_or_erode_mask(mask: Tensor, radius: int, num_iterations=1) -> Tensor:
    """
    Dilate or erode a binary mask using a square kernel.

    This function either dilates or erodes a 2D binary mask based on the given radius
    and number of iterations. A positive radius value will dilate the mask, while a
    negative radius value will erode it.

    Parameters:
    -----------
    mask : torch.Tensor
        A 2D binary mask of shape (H, W), where H is the height and W is the width.
        The dtype must be torch.bool.
    radius : int
        The radius of the square kernel used for dilation or erosion. A positive value
        will dilate the mask, while a negative value will erode it.
    num_iterations : int, optional
        The number of times the dilation or erosion operation should be applied.
        Default is 1.

    Returns:
    --------
    Tensor : torch.Tensor
        A dilated or eroded 2D binary mask of the same shape as the input mask.

    Raises:
    -------
    AssertionError
        If the dtype of the input mask is not torch.bool.

    Example:
    --------
    >>> mask = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=torch.bool)
    >>> dilated_mask = dilate_or_erode_mask(mask, radius=1)
    >>> eroded_mask = dilate_or_erode_mask(mask, radius=-1)

    """
    assert mask.dtype == torch.bool, mask.dtype
    abs_radius = abs(radius)
    erode = radius < 0
    if erode:
        mask = ~mask
    mask = mask.half()
    conv_kernel = torch.ones((1, 1, abs_radius, abs_radius), dtype=mask.dtype, device=mask.device)
    for _ in range(num_iterations):
        mask = mask.half()
        mask = F.conv2d(mask, conv_kernel, padding="same")
        mask = mask > 0.0
    if erode:
        mask = ~mask
    return mask


def get_cropped_image_with_padding(self, image, bbox, padding: float = 1.0):
    """
    Crop an image based on a bounding box with optional padding.

    Given an image and a bounding box, this function returns a cropped version of
    the image. Padding can be applied to extend the area of the cropped region.

    Parameters:
    -----------
    image : torch.Tensor
        Input image tensor of shape (C, H, W), where C is the number of channels,
        H is the height, and W is the width.
    bbox : torch.Tensor
        A bounding box tensor of shape (2, 2), where the first row contains the
        (y, x) coordinates of the top-left corner, and the second row contains the
        (y, x) coordinates of the bottom-right corner.
    padding : float, optional
        Padding factor applied to the bounding box dimensions. Default is 1.0, which
        means no padding. A value greater than 1.0 will increase the cropped area.

    Returns:
    --------
    cropped_image : torch.Tensor
        The cropped image tensor of shape (C, H', W'), where H' and W' are the
        dimensions of the cropped region.

    Example:
    --------
    >>> image = torch.rand(3, 100, 100)
    >>> bbox = torch.tensor([[10, 20], [50, 60]])
    >>> cropped_image = get_cropped_image_with_padding(image, bbox, padding=1.2)

    Notes:
    ------
    The function ensures that the cropped region does not exceed the original image
    dimensions. If the padded bounding box does, it will be clipped to fit within
    the image.

    """
    im_h = image.shape[1]
    im_w = image.shape[2]
    # bbox = iv.bbox
    x = bbox[0, 1]
    y = bbox[0, 0]
    w = bbox[1, 1] - x
    h = bbox[1, 0] - y
    x = 0 if (x - (padding - 1) * w / 2) < 0 else int(x - (padding - 1) * w / 2)
    y = 0 if (y - (padding - 1) * h / 2) < 0 else int(y - (padding - 1) * h / 2)
    y2 = im_h if y + int(h * padding) >= im_h else y + int(h * padding)
    x2 = im_w if x + int(w * padding) >= im_w else x + int(w * padding)
    cropped_image = image[
        :,
        y:y2,
        x:x2,
    ]
    return cropped_image


def interpolate_image(image: Tensor, scale_factor: float = 1.0, mode: str = "nearest"):
    """
    Interpolates images by the specified scale_factor using the specific interpolation mode.
    This method uses `torch.nn.functional.interpolate` by temporarily adding batch dimension and channel dimension for 2D inputs.
    image (Tensor): image of shape [3, H, W] or [H, W]
    scale_factor (float): multiplier for spatial size
    mode: (str): algorithm for interpolation: 'nearest' (default), 'bicubic' or other interpolation modes at https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
    """

    if len(image.shape) == 2:
        image = image.unsqueeze(0)

    image_downsampled = (
        torch.nn.functional.interpolate(
            image.unsqueeze(0).float(),
            scale_factor=scale_factor,
            mode=mode,
        )
        .squeeze()
        .squeeze()
        .bool()
    )
    return image_downsampled


def adjust_intrinsics_matrix(K, old_size, new_size):
    """
    Adjusts the camera intrinsics matrix after resizing an image.

    Args:
        K (np.ndarray): the original 3x3 intrinsics matrix.
        old_size (list[int]): the original size of the image in (width, height).
        new_size (list[int]): the new size of the image in (width, height).
    Returns:
        np.ndarray: the adjusted 3x3 intrinsics matrix.

    :example:
    >>> K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])
    >>> old_size = (640, 480)
    >>> new_size = (320, 240)
    >>> K_new = adjust_intrinsics_matrix(K, old_size, new_size)
    """
    # Calculate the scale factors for width and height
    scale_x = new_size[0] / old_size[0]
    scale_y = new_size[1] / old_size[1]

    # Adjust the intrinsics matrix
    K_new = copy.deepcopy(K)
    K_new[0, 0] *= scale_x  # Adjust f_x
    K_new[1, 1] *= scale_y  # Adjust f_y
    K_new[0, 2] *= scale_x  # Adjust c_x
    K_new[1, 2] *= scale_y  # Adjust c_y

    return K_new


def adjust_gamma(image: np.ndarray, gamma: float = 1.0):
    """Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values. Gamma correction is used to adjust the brightness of an image. Gamma = 1.0 has no effect, gamma < 1.0 darkens the image, and gamma > 1.0 brightens the image.

    Args:
        image (numpy.ndarray): The image to adjust.
        gamma (float): The gamma value to apply to the image.
    """

    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def autoAdjustments_with_convertScaleAbs(img):
    # Initial code copied from
    # https://answers.opencv.org/question/75510/how-to-make-auto-adjustmentsbrightness-and-contrast-for-image-android-opencv-image-correction/
    a_low = img.min()
    # ahigh = img.max()
    a_high = np.percentile(img, 90)
    a_max = 255
    a_min = 0

    # calculate alpha, beta
    alpha = (a_max - a_min) / (a_high - a_low)
    beta = a_min - a_low * alpha
    # perform the operation g(x,y)= α * f(x,y)+ β
    new_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # return [new_img, alpha, beta]
    return new_img


def scale_camera_matrix(K: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    Modify the camera matrix K when shrinking an image by a scale factor.

    Parameters:
    K (numpy.ndarray): 3x3 camera intrinsic matrix
    scale_factor (float): Scale factor for image shrinking (0 < scale_factor <= 1)

    Returns:
    numpy.ndarray: Modified 3x3 camera matrix
    """
    if not 0 < scale_factor <= 1:
        raise ValueError("Scale factor must be between 0 and 1")

    # Create a copy of K to avoid modifying the original matrix
    K_scaled = K.copy()

    # Scale the focal length (fx, fy) and principal point (cx, cy)
    K_scaled[0, 0] *= scale_factor  # fx
    K_scaled[1, 1] *= scale_factor  # fy
    K_scaled[0, 2] *= scale_factor  # cx
    K_scaled[1, 2] *= scale_factor  # cy

    return K_scaled


def numpy_image_to_bytes(np_image: np.ndarray) -> io.BytesIO:
    """Convert a numpy image to a byte array."""
    # Create a BytesIO object
    byte_arr = io.BytesIO()

    # Create an Image object
    image = Image.fromarray(np_image)

    # Save the image to the BytesIO object
    image.save(byte_arr, format="PNG")  # Save as PNG

    # Move the cursor to the beginning of the BytesIO object
    byte_arr.seek(0)

    return byte_arr
