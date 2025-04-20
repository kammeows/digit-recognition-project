# import numpy as np
# import cv2
# import joblib

# mnist_svm = joblib.load('models/svm_mnist_model.pkl')
# # mnist_svm_rbf = joblib.load('models/svm_mnist_model_rbf.pkl')

# image = cv2.imread('images/threeDigital.jpg', cv2.IMREAD_GRAYSCALE)
# image = cv2.resize(image, (500, 500), interpolation=cv2.INTER_LINEAR)

# blurred = cv2.GaussianBlur(image, (5,5), 0) # to get rid of noise, but there isnt a lot of it in this image
# # cv2.imshow('bllurred_Image', blurred)
# canny = cv2.Canny(blurred, 50,200,255)

# # extract the edges and make a bounding box around the digits
# # we find contours first, filter out noise from it, and then draw the bounding boxrm

# contours,_ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# boundaryBoxImage = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
# for cnt in contours:
#     x,y,w,h = cv2.boundingRect(cnt)
#     if w>10 and h>10:
#         cv2.rectangle(boundaryBoxImage, (x,y), (x+w,y+h), (0,255,0),2)

# cv2.imshow('Original image', image)
# cv2.imshow('Image with boundary box', boundaryBoxImage)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# digit_images = []
# for cnt in contours:
#     x, y, w, h = cv2.boundingRect(cnt)
#     if w > 10 and h > 10:
#         digit = image[y:y+h, x:x+w]

#         resized_digit_28x28 = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)
#         inverted_digit_28x28 = cv2.bitwise_not(resized_digit_28x28)
#         normalized_digit_28x28 = inverted_digit_28x28 / 255.0
#         flat_digit_28x28 = normalized_digit_28x28.flatten().reshape(1, -1)

#         prediction_mnist_svm = mnist_svm.predict(flat_digit_28x28)[0]
#         print("Predicted (SVM - MNIST 28x28):", prediction_mnist_svm)

#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

import numpy as np
import cv2
import joblib

mnist_svm = joblib.load('models/svm_mnist_model.pkl')
# mnist_svm = joblib.load('svm_mnist_model_rbf.pkl')

image = cv2.imread('images/two_khudka.jpg', cv2.IMREAD_GRAYSCALE)

h, w = image.shape
scale = 500 / max(h, w)
image = cv2.resize(image, (int(w*scale), int(h*scale)))

# image = cv2.resize(image, (500, 500), interpolation=cv2.INTER_LINEAR)

blurred = cv2.GaussianBlur(image, (5,5), 0) # to get rid of noise, but there isnt a lot of it in this image
# cv2.imshow('bllurred_Image', blurred)
avg_intensity = np.mean(blurred)
lower = max(0, int(0.7 * avg_intensity))
upper = min(255, int(1.3 * avg_intensity))
canny = cv2.Canny(blurred, lower, upper)

# extract the edges and make a bounding box around the digits
# we find contours first, filter out noise from it, and then draw the bounding boxrm

contours,_ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

boundaryBoxImage = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    if w>10 and h>10:
        cv2.rectangle(boundaryBoxImage, (x,y), (x+w,y+h), (0,255,0),2)

cv2.imshow('Original image', image)
cv2.imshow('Image with boundary box', boundaryBoxImage)

cv2.waitKey(0)
cv2.destroyAllWindows()

digit_images = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w > 10 and h > 10:
        # digit = image[y:y+h, x:x+w]

        # resized_digit_28x28 = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)
        # inverted_digit_28x28 = cv2.bitwise_not(resized_digit_28x28)
        # cv2.imshow("Resized Digit", inverted_digit_28x28)
        # # cv2.imshow("Inverted Digit", inverted_digit_28x28)

        # normalized_digit_28x28 = resized_digit_28x28 / 255.0
        # flat_digit_28x28 = normalized_digit_28x28.flatten().reshape(1, -1)

        digit = image[y:y+h, x:x+w]
        resized = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)
        padded = np.pad(resized, ((4,4), (4,4)), "constant", constant_values=0)  # 20x20 -> 28x28
        inverted = cv2.bitwise_not(padded)
        normalized = inverted / 255.0
        flat = normalized.flatten().reshape(1, -1)


        prediction_mnist_svm = mnist_svm.predict(flat)[0]
        print("Predicted (SVM - MNIST 28x28):", prediction_mnist_svm)

cv2.waitKey(0)
cv2.destroyAllWindows()