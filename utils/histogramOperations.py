import numpy as np
import cv2

def calcHist(image):
    r,c = image.shape
    pixelDict = {i:0 for i in range(256)} # gray level : nk
    for i in range(r):
        for j in range(c):
            pixelDict[image[i,j]] += 1
    return pixelDict

def getPDF(pixelDict):
    summ = sum(pixelDict.values())
    pdf = {i:pixelDict[i]/summ for i in range(256)}
    return pdf

def getCDF(pixelDict):
    sk = {}
    cumulative = 0    
    for i in range(256):
        cumulative += pixelDict[i]
        sk[i] = round(cumulative * 255)
    return sk

def HE(image, sk):
    r,c = image.shape
    new_image = np.zeros((r,c), dtype=np.uint8)
    for i in range(r):
        for j in range(c):
            new_image[i, j] = sk[image[i, j]]
    return new_image

def equalizeHistogram(image):
    pixelDict = calcHist(image)
    pdf = getPDF(pixelDict)
    sk = getCDF(pdf)
    equalized_image = HE(image, sk)
    equalized_image = cv2.resize(equalized_image, (500, 500), interpolation=cv2.INTER_LINEAR)
    return equalized_image

def contrast_stretching(image):
    r,c = image.shape
    result = np.zeros((r,c), dtype=np.uint8)
    Imin = 255
    Imax = 0
    Imin = np.min(image)
    Imax = np.max(image)
    meow = Imax - Imin
    for x in range(r):
        for y in range(c):
            result[x][y] = ((image[x][y] - Imin)/(meow))*255
    # contrasted_image = cv2.resize(result,(300,300),interpolation=cv2.INTER_NEAREST)
    return result