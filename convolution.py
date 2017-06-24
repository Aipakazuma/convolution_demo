# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from skimage import color


def iter_pixels(image):
    """ Yield pixel position (row, column) and pixel intensity. """
    height, width = image.shape[:2]
    for i in range(height):
        for j in range(width):
            yield (i, j), image[i, j]


def imshow_pair(image_pair, titles=('', ''), figsize=(10, 5), **kwargs):
    fig, axes = plt.subplots(ncols=2, figsize=figsize)
    for ax, img, label in zip(axes.ravel(), image_pair, titles):
        ax.imshow(img, **kwargs)
        ax.set_title(label)


def padding_for_kernel(kernel):
    """ パディングの数を返す.

    [1, 2]の値が返ってきたとしたら
    上下に1のpadding、左右に2のpaddingができることになる
    """
    # Slice to ignore RGB channels if they exist.
    image_shape = kernel.shape[:2]
    # We only handle kernels with odd dimensions so make sure that's true.
    # (The "center" pixel of an even number of pixels is arbitrary.)
    assert all((size % 2) == 1 for size in image_shape)
    return [(size - 1) // 2 for size in image_shape]


def add_padding(image, kernel):
    h_pad, w_pad = padding_for_kernel(kernel)
    # 0でpaddingを取る
    return np.pad(image, ((h_pad, h_pad), (w_pad, w_pad)),
                  mode='constant', constant_values=0)

def remove_padding(image, kernel):
    inner_region = []  # A 2D slice for grabbing the inner image region
    for pad in padding_for_kernel(kernel):
        slice_i = slice(None) if pad == 0 else slice(pad, -pad)
        inner_region.append(slice_i)
    return image[inner_region]


def window_slice(center, kernel):
    r, c = center
    r_pad, c_pad = padding_for_kernel(kernel)
    # Slicing is (inclusive, exclusive) so add 1 to the stop value
    return [slice(r - r_pad, r + r_pad + 1), slice(c - c_pad, c + c_pad + 1)]


def apply_kernel(center, kernel, original_image):
    image_patch = original_image[window_slice(center, kernel)]
    # An element-wise multiplication followed by the sum
    return np.sum(kernel * image_patch)


def iter_kernel_labels(image, kernel):
    """ Yield position and kernel labels for each pixel in the image.

    The kernel label-image has a 2 at the center and 1 for every other
    pixel "under" the kernel. Pixels not under the kernel are labeled as 0.

    Note that the mask is the same size as the input image.
    """
    original_image = image
    image = add_padding(original_image, kernel)
    i_pad, j_pad = padding_for_kernel(kernel)

    for (i, j), pixel in iter_pixels(original_image):
        # Shift the center of the kernel to ignore padded border.
        # パディング境界線を無視するようにカーネルの中心をシフトします。
        i += i_pad
        j += j_pad
        mask = np.zeros(image.shape, dtype=int)  # Background = 0
        mask[window_slice((i, j), kernel)] = 1   # Kernel = 1
        mask[i, j] = 2                           # Kernel-center = 2
        yield (i, j), mask


def visualize_kernel(kernel_labels, image):
    """ Return a composite image, where 1's are yellow and 2's are red.

    See `iter_kernel_labels` for info on the meaning of 1 and 2.
    """
    return color.label2rgb(kernel_labels, image, bg_label=0,
                           colors=('yellow', 'red'))


def make_convolution_step_function(image, kernel):
    # Initialize generator since we're only ever going to iterate over
    # a pixel once. The cached result is used, if we step back.
    gen_kernel_labels = iter_kernel_labels(image, kernel)
    image_cache = []
    image = add_padding(image, kernel)

    def convolution_step(i_step):
        """ Plot original image and kernel-overlay next to filtered image.

        For a given step, check if it's in the image cache. If not
        calculate all necessary images, then plot the requested step.
        """

        # Create all images up to the current step, unless they're already
        # cached:
        while i_step >= len(image_cache):

            # For the first step (`i_step == 0`), the original image is the
            # filtered image; after that we look in the cache, which stores
            # (`kernel_overlay`, `filtered`).
            filtered_prev = image if i_step == 0 else image_cache[-1][1]
            # We don't want to overwrite the previously filtered image:
            filtered = filtered_prev.copy()

            # Get the labels used to visualize the kernel
            center, kernel_labels = gen_kernel_labels.__next__()
            # Modify the pixel value at the kernel center
            filtered[center] = apply_kernel(center, kernel, image)
            # Take the original image and overlay our kernel visualization
            kernel_overlay = visualize_kernel(kernel_labels, image)
            # Save images for reuse.
            image_cache.append((kernel_overlay, filtered))

        # Remove padding we added to deal with boundary conditions
        # (Loop since each step has 2 images)
        image_pair = [remove_padding(each, kernel)
                      for each in image_cache[i_step]]
        return image_pair

    return convolution_step  # <-- this is a function
