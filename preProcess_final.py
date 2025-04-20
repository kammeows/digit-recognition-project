import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
# import tensorflow as tf
# print(tf.__version__)

# model = load_model('models/digit-recognition-model-2.keras')
model = load_model('models/digit-symbol-recognition.keras')
# image = cv2.imread('images/without_noise/one_two.jpg', cv2.IMREAD_GRAYSCALE)
image = cv2.imread('images/digits_symbols/fourplus6.jpg', cv2.IMREAD_GRAYSCALE)
image = cv2.GaussianBlur(image, (5,5), 0) 
# image = cv2.imread('images/with_noise/oneTwo2.jpg', cv2.IMREAD_GRAYSCALE)
# image = cv2.resize(image, (500, 500), interpolation=cv2.INTER_CUBIC)

def thresholdImage(img, Lh=100, Th=150):
    r,c = img.shape
    new_image = np.zeros(img.shape, dtype=np.uint8)
    for i in range(r):
        for j in range(c):
            if Lh<=img[i,j]<Th:
                new_image[i,j] = 1
    return new_image

binary_img = thresholdImage(image, 0, 100) # for pen/all
# binary_img = thresholdImage(image, 80, 90).astype(np.uint8) * 255 
# binary_img = thresholdImage(image, 130, 142).astype(np.uint8) * 255 # for oneTwo2

kernel = np.ones((5, 5), np.uint8)
eroded = cv2.erode(binary_img, kernel, iterations=1)
dilated = cv2.dilate(binary_img, kernel, iterations=1)
opening = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

titles = ['Original', 'Eroded', 'Dilated', 'Opening', 'Closing']
images = [binary_img, eroded, dilated, opening, closing]

and1 = cv2.bitwise_and(eroded,dilated)
and2 = cv2.bitwise_and(closing,opening)
and_image = cv2.bitwise_and(and1,and2)

# masked_img = cv2.bitwise_and(image, image, mask=and_image)
masked_img = cv2.bitwise_and(binary_img, binary_img, mask=dilated)

contours, _ = cv2.findContours(masked_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(contour_image, contours, -1, (0, 0, 255), 1)
boundaryBoxImage = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
digit_regions = []

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w > 10 and h > 10:
        digit = image[y:y+h, x:x+w]
        digit_regions.append((x, digit))
        cv2.rectangle(boundaryBoxImage, (x, y), (x+w, y+h), (0, 255, 0), 2)

# plt.imshow(masked_img, cmap='gray')
# plt.title("masked image")
# plt.axis('off')
# plt.show()

plt.imshow(boundaryBoxImage, cmap='gray')
plt.title("boundary box")
plt.axis('off')
plt.show()

# plt.imshow(and_image, cmap='gray')
# plt.title("and image")
# plt.axis('off')
# plt.show()

# plt.figure(figsize=(15, 5))
# for i in range(5):
#     plt.subplot(1, 5, i+1)
#     plt.imshow(images[i], cmap='gray')
#     plt.title(titles[i])
#     plt.axis('off')
# plt.tight_layout()
# plt.show()

def resize_and_center_digit(img, size=28, pad=4):
    h, w = img.shape
    if h > w:
        new_h = size - pad
        # new_w = int(w * (new_h / h))
        new_w = max(1, int(w * (new_h / h)))
    else:
        new_w = size - pad
        # new_h = int(h * (new_w / w))
        new_h = max(1, int(h * (new_w / w)))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((size, size), dtype=np.uint8)
    x_offset = (size - new_w) // 2
    y_offset = (size - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return canvas

digit_regions.sort(key=lambda tup: tup[0])
num_digits = len(digit_regions)
plt.figure(figsize=(num_digits * 2, 4))  # Wider based on how many digits

for i, (x, digit_img) in enumerate(digit_regions):
    # Preprocess
    processed_digit = resize_and_center_digit(digit_img)
    processed_digit = 255 - processed_digit
    processed_digit = processed_digit.reshape(1, 28, 28, 1).astype('float32') / 255.0

    # Predict
    prediction = model.predict(processed_digit)
    predicted_class = np.argmax(prediction)

    # Plot processed image
    plt.subplot(2, num_digits, i + 1)
    plt.imshow(processed_digit.squeeze(), cmap='gray')
    plt.title(f"Pred: {predicted_class}")
    plt.axis('off')

    # Plot original cropped digit
    plt.subplot(2, num_digits, num_digits + i + 1)
    plt.imshow(digit_img, cmap='gray')
    plt.title("Original")
    plt.axis('off')

plt.tight_layout()
plt.show()


# for i, (x, digit_img) in enumerate(digit_regions):
#     # method 1:
#     # prediction = model.predict(digit_img)
#     # predicted_class = np.argmax(prediction)

#     # method 2:
#     # resized_digit = cv2.resize(digit_img, (28, 28), interpolation=cv2.INTER_AREA)
#     # normalized_digit = resized_digit.astype("float32") / 255.0
#     # input_digit = np.expand_dims(normalized_digit, axis=(0, -1))
#     # prediction = model.predict(input_digit)
#     # predicted_class = np.argmax(prediction)

#     # method 3:
#     processed_digit = resize_and_center_digit(digit_img)
#     processed_digit = 255 - processed_digit
    
#     processed_digit = processed_digit.reshape(1, 28, 28, 1).astype('float32') / 255.0

#     # Predict the digit
#     prediction = model.predict(processed_digit)
#     predicted_class = np.argmax(prediction)