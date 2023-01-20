import cv2
import numpy as np


class SourceSizeGraterThanTargetSizeError(Exception):
    def __init__(self):
        msg = "Width and height of the target_size must be " \
            + "greater than that of the input_size."
        super().__init__(msg)


def fit_input_size_to_stride(input_size, stride):
    """ Fit the input_size to multiple of the stride.
    Args:
        input_size(list-like, [int, int]): width, height
        stride(int):
    Returns:
        output_size(tuple, (int, int)): width, height
    """
    def fit(x):
        if x % stride:
            x = x - (x % stride) + stride
        return x

    w, h = input_size
    output_size = fit(w), fit(h)
    return output_size


def get_resize_factors(source_size, target_size):
    """ Get resize factors for resizing source_size to target_size.
    Args:
        source_size(list-like, [int, int]): width, height
        target_size(list-like, [int, int]): width, height
    Returns:
        resize_factors(tuple, (float, float)):
    """
    w1, h1 = source_size
    w2, h2 = target_size

    rw = w2 / w1  # ratio
    rh = h2 / h1

    resize_factors = rw, rh
    return resize_factors


def get_gap_size(source_size, target_size):
    """ Get the gap_size between source_size and target_size.
    Args:
        source_size(list-like, [int, int]): width, height
        target_size(list-like, [int, int]): width, height
    Returns:
        gap_size(tuple, (int, int, int, int)): top, bottom, left, right
    Raise:
        SourceSizeGraterThanTargetSizeError
    """
    w1, h1 = source_size
    w2, h2 = target_size
    if w1 > w2 or h1 > h2:
        raise SourceSizeGraterThanTargetSizeError

    half_w = (w2 - w1) / 2
    half_h = (h2 - h1) / 2

    top = int(round(half_h - 0.1)) 
    bottom = int(round(half_h + 0.1))
    left = int(round(half_w - 0.1))
    right = int(round(half_w + 0.1))

    gap_size = top, bottom, left, right
    return gap_size


def add_border_lines(mat, thickness, rgb=(0,0,0)):
    """ Add border_lines to the image.
    Args:
        mat(np.ndarray): image matrix
        thickness(list-like, [int, int, int, int]): top, bottom, left, right
        rgb(list-like, [int, int, int]):
    Returns:
        bordered_mat(np.ndarray): bordered image matrix
    """
    border_type = cv2.BORDER_CONSTANT
    top, bottom, left, right = thickness

    bordered_mat = cv2.copyMakeBorder(
        mat,
        top, bottom, left, right,
        border_type, value=rgb,)
    return bordered_mat


def resize_img(mat, resize_factors):
    """ Resize the image with resize_factors.
    Args:
        mat(np.ndarray): image matrix
        resize_factors(list-like, [float, float]):
    Returns:
        target_mat(np.ndarray):
    """
    source_size = np.array(mat.shape[:2][::-1])
    target_size = np.round(source_size * resize_factors).astype(int)

    target_mat = cv2.resize(mat, target_size)
    return target_mat


def resize_img_keeping_aspect_ratio(mat, target_size, stride=None):
    """ Resize the image to the target_size with keeping aspect ratio.
    Args:
        mat(np.ndarray): image matrix
        target_size(list-like, [int, int]): width, height
        stride(int):
    Returns:
        resized_mat(np.ndarray): resized image matrix
        resize_info(dict):
    """
    if stride is not None:
        target_size = fit_input_size_to_stride(target_size, stride)

    source_size = mat.shape[:2][::-1]
    resize_factors = get_resize_factors(source_size, target_size)
    resize_factors = (min(resize_factors),) * 2  # (min, min)
    resized_mat = resize_img(mat, resize_factors)
    
    source_size = resized_mat.shape[:2][::-1]
    gap_size = get_gap_size(source_size, target_size)
    diff_origin = gap_size[2], gap_size[0]  # (top, left)
    resized_mat = add_border_lines(resized_mat, gap_size)

    resize_info = {
        "resize_factors": resize_factors, "diff_origin": diff_origin}
    return resized_mat, resize_info
