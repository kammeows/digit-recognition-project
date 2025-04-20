# from tensorflow.keras.models import load_model
# model = load_model('models/tf-cnn-model.h5', compile=False)
# model.save('saved_tfModel_opencv.h5')  # This creates a folder with .pb file

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

model = load_model('models/saved_tfModel_opencv.h5')

def preprocess_digit(img):
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    if np.mean(img) > 127:
        img = 255 - img
    img = img.astype('float32') / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img

# image = cv2.imread('images/without_noise/one2345.jpg', cv2.IMREAD_GRAYSCALE)
image = cv2.imread('images/with_noise/oneTwo.jpg', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (500, 500), interpolation=cv2.INTER_LINEAR)

blurred = cv2.GaussianBlur(image, (5, 5), 0)
canny = cv2.Canny(blurred, 50, 200)

contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

boundaryBoxImage = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
digit_regions = []

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w > 10 and h > 10:
        digit = image[y:y+h, x:x+w]
        digit_regions.append((x, digit))  # Save with x for sorting
        cv2.rectangle(boundaryBoxImage, (x, y), (x+w, y+h), (0, 255, 0), 2)

plt.imshow(boundaryBoxImage)
plt.title("Detected Digits")
plt.axis('off')
plt.show()

digit_regions.sort(key=lambda tup: tup[0])

for i, (x, digit_img) in enumerate(digit_regions):
    input_digit = preprocess_digit(digit_img)
    prediction = model.predict(input_digit)
    predicted_class = np.argmax(prediction)

    print(f"Digit {i+1} (x={x}): Predicted = {predicted_class}")

    plt.subplot(1, len(digit_regions), i+1)
    plt.imshow(digit_img, cmap='gray')
    plt.title(str(predicted_class))
    plt.axis('off')

plt.tight_layout()
plt.show()
