import cv2
import numpy as np
import matplotlib.pyplot as plt
from kmeansCustom import KMeans


def segment_image(image, y_pred):
    labels = y_pred.flatten()
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)

    return segmented_image, labels


def create_masked_image(labels, image):
    BLUE = (0, 0, 255)
    RED = (255, 0, 0)
    masked_image = np.copy(image)
    masked_image = masked_image.reshape((-1, 3))
    cluster = 2
    masked_image[labels == cluster] = [0, 0, 255]
    masked_image = masked_image.reshape(image.shape)

    # Find upper and lower color from BGR to HSV
    hsv_img = cv2.cvtColor(masked_image, cv2.COLOR_RGB2HSV)
    blue = np.uint8([[[255, 0, 0]]])
    # hsv_blue = cv2.cvtColor(blue, cv2.COLOR_BGR2HSV)
    return hsv_img

# # --------------------------------- ROI of Tumor --------------------------------------


def detect_tumor(image, hsv_img):
    lower_blue = (120, 255, 250)
    upper_blue = (120, 255, 255)
    COLOR_MIN = np.array([lower_blue], np.uint8)
    COLOR_MAX = np.array([upper_blue], np.uint8)
    frame_threshed = cv2.inRange(hsv_img, COLOR_MIN, COLOR_MAX)
    imgray = frame_threshed
    ret, thresh = cv2.threshold(frame_threshed, 127, 255, 0)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the index of the largest contour
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]

    x, y, w, h = cv2.boundingRect(cnt)

    return x, y, w, h


# Testing
if __name__ == "__main__":
    # # input image path
    image = cv2.imread("./image(120).jpg")
    # BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # KMeans
    k = KMeans(K=3, max_iters=100)
    y_pred = k.predict(pixel_values)
    centers = np.uint8(k.cent())
    y_pred = y_pred.astype(int)

    # Segmented image
    segment_img, labels = segment_image(image, y_pred)

    # Create masked image of hsv
    hsv_img = create_masked_image(labels, image)

    # Detect Tumor
    x, y, w, h = detect_tumor(image, hsv_img)

    pad_w = 3
    pad_h = 4
    pad_x = 3
    pad_y = 4

    cv2.rectangle(image, (x-pad_x, y-pad_y),
                  (x+w+pad_w, y+h+pad_h), (255, 0, 0), 2)
    plt.imshow(image)
    plt.savefig('tumor.jpg')
