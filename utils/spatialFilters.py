# adaptive median and mean filter
# unsharp masking
# log
# dog

import numpy as np
import cv2
import matplotlib.pyplot as plt

def resize_digit(img):
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    if np.mean(img) > 127:
        img = 255 - img
    img = img.astype('float32') / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.5, threshold=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount+1)*image - float(amount)*blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)

    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)

    return sharpened

# def adaptive_mean_filter(img, kernel_size=(3,3), max_kernel_size=(7,7)):
#     pad_h = kernel_size[0] // 2
#     pad_w = kernel_size[1] // 2

#     padded_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
#     new_img = np.zeros(img.shape, dtype=np.uint8)

#     rows, cols = img.shape

#     for i in range(rows):
#         for j in range(cols):
#             # Extract kernel window from padded image
#             window = padded_img[i:i + kernel_size[0], j:j + kernel_size[1]]
#             mean_val = np.mean(window)
#             new_img[i, j] = np.uint8(mean_val)

#     return new_img


def adaptive_mean_filter(img, base_kernel=(3, 3), max_kernel=(7, 7), var_thresh=100):
    h, w = img.shape
    pad = max_kernel[0] // 2
    padded_img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    filtered_img = np.zeros_like(img)

    for i in range(pad, h + pad):
        for j in range(pad, w + pad):
            window = padded_img[i - pad:i + pad + 1, j - pad:j + pad + 1]
            local_var = np.var(window)

            if local_var > var_thresh:
                kernel_size = base_kernel  # less smoothing
            else:
                kernel_size = max_kernel  # more smoothing

            k = kernel_size[0] // 2
            region = padded_img[i - k:i + k + 1, j - k:j + k + 1]
            mean_val = np.mean(region)
            filtered_img[i - pad, j - pad] = np.uint8(mean_val)

    return filtered_img


# def adaptive_mean_filter(img, kernel_size=(3,3), max_kernel_size=(7,7)):
#     new_img = np.zeros(img.shape, dtype=np.uint8)
#     img,pad_h,pad_w = pad_image(img, kernel_size)
#     for i in range(pad_h, pad_h+img.shape[0]):
#         for j in range(pad_w, pad_w+img.shape[1]):
#             window = img[i-pad_h:i+pad_h+1, j-pad_w:j+pad_w+1]
#             mean_val = np.mean(window)
#             new_img[i-pad_h, j-pad_w] = np.uint8(mean_val)
#     return new_img