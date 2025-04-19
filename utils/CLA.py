import cv2
import matplotlib.pyplot as plt
import numpy as np

# image = cv2.imread('../images/six789_white.jpg', cv2.IMREAD_GRAYSCALE)
# image = cv2.resize(image, (290, 290), interpolation=cv2.INTER_LINEAR)

def threshold(img):
    r,c = img.shape
    new_image = np.zeros(img.shape, dtype=np.uint8)
    for i in range(r):
        for j in range(c):
            if 75<img[i,j]<80:
                new_image[i,j] = 1
    return new_image

def get_neighbours(img, pos):
    x,y = pos
    r,c = img.shape
    # neighbouring_pixels = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    neighbouring_pixels = [(-1, -1), (-1, 0), (-1, 1), (0, -1)] # top, left, right, bottom
    # neighbouring_pixels = [(x-1,y-1),(x-1,y),(x+1,y+1),(x-1,y),(x+1,y),(x-1,y-1),(x,y-1),(x+1, y-1)] # top, left, right, bottom
    neighbours = []
    for dx,dy in neighbouring_pixels:
        nx,ny = dx+x, dy+y
        if 0 <= nx < r and 0 <= ny < c:
            neighbours.append((nx,ny))
    print("the neighbours are: ", neighbours)
    return neighbours

def extract_components(original_image, labeled_image):
    components = []
    unique_labels = np.unique(labeled_image)
    unique_labels = unique_labels[unique_labels != 0]

    for label in unique_labels:
        component_mask = (labeled_image == label).astype(np.uint8)
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            cropped = original_image[y:y+h, x:x+w]
            components.append((cropped, (x, y, w, h)))  # Also return position if needed
    print("the components are: ", components)
    return components

def CLA(img):
    image = img
    # img = threshold(img)
    labeled_image = np.zeros(img.shape, dtype=np.uint32)
    label = 1 # bg = 0 and unlabeled = 1
    parent = {}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    r,c = img.shape
    for i in range(r):
        for j in range(c):
            if img[i,j] == 0:
                continue

            neighbour_labels = []
            for x,y in get_neighbours(img, (i,j)):
                if labeled_image[x,y] > 0:
                    neighbour_labels.append(labeled_image[x,y])

            if neighbour_labels == []:
                label += 1
                labeled_image[i, j] = label
                parent[label] = label
            else:
                min_label = min(neighbour_labels)
                labeled_image[i, j] = min_label
                for l in neighbour_labels:
                    root1 = find(min_label)
                    root2 = find(l)
                    if root1 != root2:
                        parent[root2] = root1  # Union
    
    for i in range(r):
        for j in range(c):
            if labeled_image[i, j] > 0:
                labeled_image[i, j] = find(int(labeled_image[i, j]))
    
    components = extract_components(image, labeled_image)

    if not components:
        print("No components extracted!")

    plt.imshow(labeled_image, cmap="gray")
    plt.title("Labeled Image")
    plt.axis('off')
    plt.show()

    plt.imshow(image, cmap='gray')
    plt.title("original Image")
    plt.axis('off')
    plt.show()

    thresh_img = threshold(image)
    plt.imshow(thresh_img, cmap='gray')
    plt.title("Thresholded")
    plt.show()

    for i, (comp, _) in enumerate(components):
        plt.subplot(1, len(components), i+1)
        plt.imshow(comp, cmap='gray')
        plt.axis('off')
    plt.show()

    return labeled_image

# def CLA(img):
#     img = threshold(img)
#     print("Unique pixel values:", np.unique(img))
#     labeled_image = np.zeros(img.shape, dtype=np.uint32)
#     label = 2 # bg = 0 and unlabeled = 1
#     equivalences = {}

#     r,c = img.shape

#     for i in range(r):
#         for j in range(c):
#             if img[i,j] == 0:
#                 print("skipped")
#                 continue

#             neighbour_labels = []
#             for x,y in get_neighbours(img, (i,j)):
#                 if labeled_image[x,y] > 0:
#                     neighbour_labels.append(labeled_image[x,y])

#             if neighbour_labels == []:
#                 label += 1
#                 labeled_image[i,j] = label
#                 equivalences[label] = label
#             else:
#                 min_label = min(neighbour_labels)
#                 labeled_image[i, j] = min_label
#             print('nl are',neighbour_labels)

#     plt.imshow(labeled_image, cmap='nipy_spectral')
#     plt.title("tyfj Image")
#     plt.axis('off')
#     plt.show()
#     return labeled_image

# labeled_image = CLA(image)
# print("Unique labels found:", np.unique(labeled_image))