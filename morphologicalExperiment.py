import cv2
import numpy as np
import matplotlib.pyplot as plt

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

def thresholdd(img, Lh=100, Th=150):
    r,c = img.shape
    new_image = np.zeros(img.shape, dtype=np.uint8)
    for i in range(r):
        for j in range(c):
            if Lh<=img[i,j]<Th:
                new_image[i,j] = 1
    return new_image

image = cv2.imread("images/pen/one23.jpg", cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (500, 500), interpolation=cv2.INTER_LINEAR)
# adaptiveMeanFilter = adaptive_mean_filter(image)
# _, binary_img = cv2.threshold(adaptiveMeanFilter, 80, 90, cv2.THRESH_BINARY)
binary_img = thresholdd(image, 0, 100)


kernel = np.ones((5, 5), np.uint8)  # You can change size (e.g., (5, 5)) or shape

eroded = cv2.erode(binary_img, kernel, iterations=1)

dilated = cv2.dilate(binary_img, kernel, iterations=1)

opening = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)

closing = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)


titles = ['Original', 'Eroded', 'Dilated', 'Opening', 'Closing']
images = [binary_img, eroded, dilated, opening, closing]

# Assume binary_img and opening are two binary images
# xnor = cv2.bitwise_not(cv2.bitwise_xor(eroded, dilated))
andd1 = cv2.bitwise_and(eroded,dilated)
andd2 = cv2.bitwise_and(closing,opening)
andd = cv2.bitwise_and(andd1,andd2)

plt.imshow(andd, cmap='gray')
plt.title("andd")
plt.axis('off')
plt.show()

plt.figure(figsize=(15, 5))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()

# Calculate histogram
hist = cv2.calcHist([image], [0], None, [256], [0, 256])

# Plot the histogram
plt.plot(hist, color='black')
plt.title("Grayscale Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


import cv2
import numpy as np
# from skimage.util import view_as_windows

def adaptive_opening(img, base_kernel=(3, 3), max_kernel=(7, 7), step=2):
    img_out = np.zeros_like(img)
    h, w = img.shape
    pad = max_kernel[0] // 2

    # Pad image
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)

    for i in range(pad, h + pad):
        for j in range(pad, w + pad):
            window = padded[i - pad:i + pad + 1, j - pad:j + pad + 1]
            var = np.var(window)

            if var < 100:
                ksize = base_kernel
            else:
                ksize = max_kernel

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
            roi = padded[i - ksize[0]//2:i + ksize[0]//2 + 1, j - ksize[1]//2:j + ksize[1]//2 + 1]
            opened = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel)
            img_out[i - pad, j - pad] = opened[ksize[0]//2, ksize[1]//2]

    return img_out

# adaptive_opening_image = adaptive_opening(image)

# plt.imshow(adaptive_opening_image, cmap='gray')
# plt.title("andd")
# plt.axis('off')
# plt.show()

# adaptiveMeanFilter = adaptive_mean_filter(image)

plt.imshow(image, cmap='gray')
plt.title("adaptivemean")
plt.axis('off')
plt.show()