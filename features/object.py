import cv2
import numpy as np
image = cv2.imread('../dataset/test/1_horse/horse_04.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
max_area = 0
main_contour = None

for contour in contours:
    area = cv2.contourArea(contour)
    if area > max_area:
        max_area = area
        main_contour = contour
x, y, w, h = cv2.boundingRect(main_contour)
cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow('Bounding Box', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

