import cv2
# Load and preprocess the image
image_path = '../dataset/train/3_rooster/rooster_005.jpg'

# Load the image
image = cv2.imread(image_path, 0)

sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(image, None)
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)

print(descriptors.shape)

cv2.imshow("Image with Keypoints", image_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
