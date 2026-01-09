import cv2

# Load image
img = cv2.imread("input_photo/input.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 1: Smooth image
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Step 2: Edge detection
edges = cv2.Canny(blur, 50, 150)

# Step 3: Thresholding
_, binary = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)

# Display results
cv2.imshow("Original", img)
cv2.imshow("Edges", edges)
cv2.imshow("Binary Output", binary)

# Save result
cv2.imwrite("saved_photo/final_binary_image.jpg", binary)

cv2.waitKey(0)
cv2.destroyAllWindows()
