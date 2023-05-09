from PIL import Image

# Load the image as a PIL image
image = Image.open("dataset/train/10_flower/flower_001.jpg")

# Convert the image to grayscale
gray = image.convert('L')

# Set the threshold value
threshold = 128

# Create a binary image by thresholding the grayscale image
binary = gray.point(lambda x: 0 if x < threshold else 255, '1')

print(binary.size)

# Display the binary image
binary.show()
