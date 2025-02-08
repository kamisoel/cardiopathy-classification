from monai.config import KeysCollection, DtypeLike
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_obj import get_track_meta
from monai.utils.type_conversion import convert_to_dst_type, convert_to_tensor, convert_data_type
from monai.networks.layers import apply_filter
from monai.utils.enums import TransformBackends
from monai.utils import ensure_tuple
from typing import Sequence, Mapping, Hashable, Union, Optional, Callable
from monai import transforms
from monai.transforms.utils import is_positive
import torch
import numpy as np
import monai.transforms.spatial.functional as func_transforms

from skimage import exposure
from scipy.linalg import eigh

# adapted from monai.transforms.MaskIntensity source code
class MultiMaskIntensity(transforms.Transform):
    """
    Mask the intensity values of input image with the specified mask data.
    Mask data must have the same spatial size as the input image, and all
    the intensity values of input image corresponding to the selected values
    in the mask data will keep the original value, others will be set to `0`.

    Args:
        mask_data: if `mask_data` is single channel, apply to every channel
            of input image. if multiple channels, the number of channels must
            match the input data. the intensity values of input image corresponding
            to the selected values in the mask data will keep the original value,
            others will be set to `0`. if None, must specify the `mask_data` at runtime.
        select_fn: function to select valid values of the `mask_data`, default is
            to select `values > 0`.

    """
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, mask_data: NdarrayOrTensor | None = None, select_labels: Sequence[int] | int = 1) -> None:
        self.mask_data = mask_data
        self.select_labels = ensure_tuple(select_labels)

    def __call__(self, img: NdarrayOrTensor, mask_data: NdarrayOrTensor | None = None) -> NdarrayOrTensor:
        """
        Args:
            mask_data: if mask data is single channel, apply to every channel
            of input image. if multiple channels, the number of channels must
            match the input data. the intensity values of input image corresponding
            to the selected values in the mask data will keep the original value,
            others will be set to `0`. if None, must specify the `mask_data` at runtime.
        select_labels: labels in `mask_data` to compare to, default is
            to select `values == 1`. if list is given, one output channel is created
            per label

        Raises:
            - ValueError: When both ``mask_data`` and ``self.mask_data`` are None.

        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        mask_data = self.mask_data if mask_data is None else mask_data
        if mask_data is None:
            raise ValueError("must provide the mask_data when initializing the transform or at runtime.")

        mask_data_, *_ = convert_to_dst_type(src=mask_data, dst=img)

        masked_list = []
        for label in self.select_labels:
            masked_list.append(img * (mask_data_ == label))
        masked_img = torch.concat(masked_list, dim=0)

        return convert_to_dst_type(masked_img, dst=img)[0]


class MultiMaskIntensityd(transforms.MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.MaskIntensity`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        mask_data: if mask data is single channel, apply to every channel
            of input image. if multiple channels, the channel number must
            match input data. the intensity values of input image corresponding
            to the selected values in the mask data will keep the original value,
            others will be set to `0`. if None, will extract the mask data from
            input data based on `mask_key`.
        mask_key: the key to extract mask data from input dictionary, only works
            when `mask_data` is None.
        select_labels: labels in `mask_data` to compare to, default is
            to select `values == 1`. if list is given, one output channel is created
            per label.
        allow_missing_keys: don't raise exception if key is missing.

    """

    backend = MultiMaskIntensity.backend

    def __init__(
        self,
        keys: KeysCollection,
        mask_data: NdarrayOrTensor | None = None,
        mask_key: str | None = None,
        select_labels: Sequence | int = 1,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.converter = MultiMaskIntensity(mask_data=mask_data, select_labels=select_labels)
        self.mask_key = mask_key if mask_data is None else None

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key], d[self.mask_key]) if self.mask_key is not None else self.converter(d[key])
        return d


class AdaptiveHistogramNormalize(transforms.Transform):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    CLAHE: An algorithm for local contrast enhancement, that uses histograms computed over different tile regions of the image.
    Local details can therefore be enhanced even in regions that are darker or lighter than most of the image.
    Refer to: https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.equalize_adapthist

    Args:
    kernel_size : int or array_like, optional
        Defines the shape of contextual regions used in the algorithm. If
        iterable is passed, it must have the same number of elements as
        ``image.ndim`` (without color channel). If integer, it is broadcasted
        to each `image` dimension. By default, ``kernel_size`` is 1/8 of
        ``image`` height by 1/8 of its width.
    clip_limit : float, optional
        Clipping limit, normalized between 0 and 1 (higher values give more
        contrast).
    num_bins : int, optional
        Number of gray bins for histogram ("data range").
    dtype: data type of the output, if None, same as input image. default to `float32`.
    """

    backend = [TransformBackends.NUMPY]

    def __init__(
        self,
        kernel_size : Optional[Union[np.ndarray, int]] = None,
        clip_limit: float = 0.01,
        num_bins: int = 256,
        dtype: DtypeLike = np.float32,
    ) -> None:
        self.clip_limit = clip_limit
        self.kernel_size = kernel_size
        self.num_bins = num_bins
        self.dtype = dtype

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        img = convert_to_tensor(img, track_meta=get_track_meta())
        img_np, *_ = convert_data_type(img, np.ndarray)

        ret = exposure.equalize_adapthist(img_np.astype(np.uint16), self.kernel_size, self.clip_limit, self.num_bins)
        out, *_ = convert_to_dst_type(src=ret, dst=img, dtype=self.dtype or img.dtype)

        return out


class AdaptiveHistogramNormalized(transforms.MapTransform):
    """
    Dictionary-based wrapper of :py:class:`custom_transforms.AdaptiveHistogramNormalize`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        kernel_size : int or array_like, optional
            Defines the shape of contextual regions used in the algorithm. If
            iterable is passed, it must have the same number of elements as
            ``image.ndim`` (without color channel). If integer, it is broadcasted
            to each `image` dimension. By default, ``kernel_size`` is 1/8 of
            ``image`` height by 1/8 of its width.
        clip_limit : float, optional
            Clipping limit, normalized between 0 and 1 (higher values give more
            contrast).
        num_bins : int, optional
            Number of gray bins for histogram ("data range").
        dtype: data type of the output, if None, same as input image. default to `float32`.
        allow_missing_keys: do not raise exception if key is missing.
    """

    backend = AdaptiveHistogramNormalize.backend

    def __init__(
        self,
        keys: KeysCollection,
        kernel_size : Optional[Union[np.ndarray, int]] = None,
        clip_limit: float = 0.01,
        num_bins: int = 256,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.transform = AdaptiveHistogramNormalize(kernel_size, clip_limit, num_bins, dtype)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.transform(d[key])

        return d

class EllipseFitRotated(transforms.MapTransform):
    """
    Transform an image by fitting an ellipse to its contours and rotating
    the image by the negative ellipse rotation. Dictionary-based implementation.
    First finds the contour of the binary mask image that only compose of 0 and 1,
    with Laplacian kernel for edge detection.

    Args:
        keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
        mask_key: data source to extract the contour, with shape: [channels, height, width[, depth]]
        select_fn: function to select expected foreground, default is to select values > 0.
        depth_axis:
        rot90_only: bool, if true only rotate by 90 degree steps, else rotate by exact theta
        target_rot: rotation in radians, if rot90_only is False, images will be rotated so the long-axis \
                    rotation is equal to target_rot (default 45°)
        allow_missing_keys: don't raise exception if key is missing.
    """
    # numpy for opencv, torch for own impl
    backend = [TransformBackends.NUMPY, TransformBackends.TORCH]

    def __init__(
        self,
        keys: KeysCollection,
        mask_key: str,
        select_fn: Callable = is_positive,
        depth_axis: int = -1,
        rot90_only: bool = True,
        target_rot: float = np.deg2rad(45),
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.kernel_type = "Laplace"
        self.keys = keys
        self.mask_key = mask_key
        self.select_fn = select_fn
        self.depth_axis = depth_axis
        self.rot90_only = rot90_only
        self.target_rot = target_rot

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        mask = d[self.mask_key]
        mask = convert_to_tensor(mask, track_meta=get_track_meta())

        ## own implementation, not working yet :(
        # fit ellipse to contours
        # contours = self._find_contour(mask)
        # ellipses = self._fit_ellipse_ams_batch(contours)
        # thetas = self._extract_rotation(ellipses)

        ## use opencv
        mask_np, *_ = convert_data_type(mask, np.ndarray)
        thetas = self._get_long_axis_rotation(mask)


        thetas = convert_to_tensor(thetas, track_meta=get_track_meta()) # (channel, depth)
        thetas = torch.nanmean(thetas, dim=1) # (channel,)
        theta = torch.nanmean(thetas) # (1,)

        for key in self.key_iterator(d):
            if self.rot90_only:
                k = -1 if (torch.rad2deg(theta) > 90) else 0
                spatial_axis = [0, 1, 2]
                del spatial_axis[self.depth_axis]
                d[key] = transforms.Rotate90(k, tuple(spatial_axis))(d[key])
            else:
                angles = torch.zeros(3)
                angles[self.depth_axis] = theta + self.target_rot
                d[key] = transforms.Rotate(angles)(d[key])
        return d


    def _get_long_axis_rotation(self, masks):
        import cv2
        masks = masks.astype(np.uint8)
        masks = np.moveaxis(masks, self.depth_axis+1, 1)
        C, D, W, H = masks.shape
        theta_batch = np.zeros((C,D), dtype=np.float32)
        unique_labels = np.unique(masks)

        for c in range(C):
            for d in range(D):
                # Extract non-zero points for the current mask slice
                mask = masks[c, d]
                # check if all segmentations are available in the slice
                if not np.isin(unique_labels, np.unique(mask)).all():
                    theta_batch[c, d] = np.nan
                    continue
                # convert to binary mask and cast to uint8
                mask = self.select_fn(mask).astype(np.uint8)
                # find outer contour
                cntrs = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]
                # get rotated rectangle from outer contour by fitting an ellipse
                if len(cntrs[0]) < 10: # not enough points to fit ellipse
                    theta_batch[c, d] = np.nan
                    continue
                ellipse = cv2.fitEllipse(cntrs[0])
                # get angle from rotated rectangle
                theta = ellipse[-1]
                theta = (180 - theta)
                theta_batch[c, d] = np.deg2rad(theta)
        return theta_batch


    def _find_contour(self, mask):
        mask = self.select_fn(mask).float()
        spatial_dims = len(mask.shape) - 1
        mask = mask.unsqueeze(0)  # add a batch dim
        if spatial_dims == 2:
            kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32)
        elif spatial_dims == 3:
            kernel = -1.0 * torch.ones(3, 3, 3, dtype=torch.float32)
            kernel[1, 1, 1] = 26.0
        else:
            raise ValueError(f"{self.__class__} can only handle 2D or 3D images.")
        contour = apply_filter(mask, kernel)
        contour.clamp_(min=0.0, max=1.0)
        return contour

    ## FIXME
    def _fit_ellipse_ams_batch(self, contour_masks):
        """
        Fit ellipses for a batch of contour masks using the AMS method.
        Args:
            contour_masks: Tensor of shape (N, C, W, H), where each slice is a contour mask.
        Returns:
            A_batch: Tensor of fitted ellipse coefficients of shape (N, C, 6).
            theta_batch: Tensor of rotation angles in radians, shape (N, C).
        """
        if contour_masks.ndim > 4:
            contour_masks = contour_masks.view(-1, * contour_masks.shape[2:])  # squeeze batch and channel dim together
        N, C, W, H = contour_masks.shape

        # Initialize output tensors
        A_batch = torch.zeros((N, C, 6), device=contour_masks.device)

        for n in range(N):
            for c in range(C):
                # Extract non-zero points for the current mask slice
                mask = contour_masks[n, c]
                x, y = mask.nonzero(as_tuple=True)
                x, y = x.float(), y.float()

                if len(x) < 6:  # Not enough points to fit an ellipse
                    continue

                # Design matrices
                D = torch.stack([x**2, x*y, y**2, x, y, torch.ones_like(x)], dim=1)
                Dx = torch.stack([2*x, y, torch.zeros_like(x), torch.ones_like(x), torch.zeros_like(x), torch.zeros_like(x)], dim=1)
                Dy = torch.stack([torch.zeros_like(x), x, 2*y, torch.zeros_like(x), torch.ones_like(x), torch.zeros_like(x)], dim=1)

                # Compute M and N
                M = D.T @ D
                Nx = Dx.T @ Dx
                Ny = Dy.T @ Dy
                N = Nx + Ny

                # Regularization
                epsilon = 1e-6
                N += epsilon * torch.eye(N.shape[0], device=contour_masks.device)

                # Solve the generalized eigenvalue problem
                eigenvalues, eigenvectors = torch.linalg.eig(torch.linalg.solve(N, M))
                eigenvalues = eigenvalues.real
                eigenvectors = eigenvectors.real

                # Extract the eigenvector corresponding to the smallest eigenvalue
                min_eigenvalue_index = torch.argmin(eigenvalues)
                A = eigenvectors[:, min_eigenvalue_index]

                # Store the coefficients
                A_batch[n, c] = A
        return A_batch


    def _extract_rotation(A):
        """
        Extract the rotation angle of the ellipse from the fitted coefficients in PyTorch.
        Args:
            A: Tensor of ellipse coefficients [..., 6].
        Returns:
            thetas: Rotation angles of the ellipses in radians.
        """
        A_xx = A[..., 0]
        A_xy = A[..., 1]
        A_yy = A[..., 2]
        # Compute rotation angle theta
        theta = 0.5 * torch.atan2(A_xy, A_xx - A_yy)
        return theta
