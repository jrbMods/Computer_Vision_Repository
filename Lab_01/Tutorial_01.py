import cv2

# Load image
img = cv2.imread("input_photo/input.jpg")

if img is None:
    print("Error loading image")
    exit()

# Display image
cv2.imshow("Original Image", img)

# Save image
cv2.imwrite("saved_photo/saved_image.jpg", img)

# Wait and close
cv2.waitKey(0)
cv2.destroyAllWindows()
