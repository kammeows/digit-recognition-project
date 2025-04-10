import numpy as np
import cv2
import joblib

knn = joblib.load('models/knn_model.pkl')
svm = joblib.load('models/svm_model.pkl')

image = cv2.imread('images/threeDigital.jpg', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (500, 500), interpolation=cv2.INTER_LINEAR)

blurred = cv2.GaussianBlur(image, (5,5), 0) # to get rid of noise, but there isnt a lot of it in this image
# cv2.imshow('bllurred_Image', blurred)
canny = cv2.Canny(blurred, 50,200,255)

# extract the edges and make a bounding box around the digits
# we find contours first, filter out noise from it, and then draw the bounding boxrm

contours,_ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

boundaryBoxImage = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    if w>10 and h>10:
        cv2.rectangle(boundaryBoxImage, (x,y), (x+w,y+h), (0,255,0),2)

cv2.imshow('Original image', image)
# cv2.imshow('Edged image', canny)
cv2.imshow('Image with boundary box', boundaryBoxImage)

cv2.waitKey(0)
cv2.destroyAllWindows()

digit_images = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w > 10 and h > 10:
        digit = image[y:y+h, x:x+w]
        resized_digit = cv2.resize(digit, (8, 8), interpolation=cv2.INTER_AREA)
        inverted_digit = cv2.bitwise_not(resized_digit)
        scaled_digit = (inverted_digit / 255.0) * 16
        flat_digit = scaled_digit.flatten().reshape(1, -1)
        
        prediction = knn.predict(flat_digit)[0]
        print("the digit is (knn): ", prediction)

        prediction = svm.predict(flat_digit)[0]
        print("the digit is (svm): ", prediction)
        cv2.waitKey(0)
        cv2.destroyAllWindows()