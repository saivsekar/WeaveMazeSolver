import cv2
import numpy as np

image = cv2.imread('maze.png', cv2.IMREAD_UNCHANGED);
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# getting mask with connectComponents
ret, labels = cv2.connectedComponents(binary)
for label in range(1,ret):
    mask = np.array(labels, dtype=np.uint8)
    mask[labels == label] = 200
    cv2.imshow('component',mask)
    cv2.waitKey(70)