import cv2

img = cv2.imread("input_photo/input.jpg")

# Gaussian Blur
gaussian = cv2.GaussianBlur(img, (15, 15), 0)

# Median Blur
median = cv2.medianBlur(img, 5)

# Bilateral Filter
bilateral = cv2.bilateralFilter(img, 9, 75, 75)

cv2.imshow("Gaussian Blur", gaussian)
cv2.imshow("Median Blur", median)
cv2.imshow("Bilateral Filter", bilateral)

cv2.waitKey(0)
cv2.destroyAllWindows()
