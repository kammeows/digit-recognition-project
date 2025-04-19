import numpy as np
import cv2
import matplotlib.pyplot as plt

def power_law_transform(image, gamma=1.0, c=1):
    image = image / 255.0
    transformed = c * np.power(image, gamma)
    transformed = np.uint8(transformed * 255)
    return transformed

def threshold(img, Lh=100, Th=150):
    r,c = img.shape
    new_image = np.zeros(img.shape, dtype=np.uint8)
    for i in range(r):
        for j in range(c):
            if Lh<=img[i,j]<Th:
                new_image[i,j] = 1
    return new_image

# # Load grayscale image
# img = cv2.imread('../images/with_noise/oneTwo2.jpg', cv2.IMREAD_GRAYSCALE)

# # Try different gamma values
# gamma_corrected = power_law_transform(img, gamma=0.4)  # Brighten
# gamma_corrected_dark = power_law_transform(img, gamma=2.0)  # Darken

# # Plot original vs transformed
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 3, 1)
# plt.imshow(img, cmap='gray')
# plt.title('Original')

# plt.subplot(1, 3, 2)
# plt.imshow(gamma_corrected, cmap='gray')
# plt.title('Gamma = 0.4 (Brightened)')

# plt.subplot(1, 3, 3)
# plt.imshow(gamma_corrected_dark, cmap='gray')
# plt.title('Gamma = 2.0 (Darkened)')

# plt.tight_layout()
# plt.show()
