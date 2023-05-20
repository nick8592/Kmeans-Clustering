import cv2

# Load and preprocess the image
image_path = '../dataset/train/5_sailboat/sailboat_002.jpg'

# Load the image
image = cv2.imread(image_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 150)
cv2.imshow("Edge", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
print(len(contours))
perimeter = cv2.arcLength(contours[0], True)
print("Perimeter:", perimeter)
cv2.drawContours(image, [contours[0]], -1, (0, 255, 0), 2)
M = cv2.moments(contours[0])
centroid_x = int(M["m10"] / M["m00"])
centroid_y = int(M["m01"] / M["m00"])
cv2.circle(image, (centroid_x, centroid_y), 5, (0, 0, 255), -1)
cv2.imshow("Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()



