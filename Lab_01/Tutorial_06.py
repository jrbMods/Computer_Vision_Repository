import cv2

img = cv2.imread("input_photo/input.jpg", cv2.IMREAD_GRAYSCALE)

_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

cv2.imshow("Binary Image", binary)
cv2.waitKey(0)
cv2.destroyAllWindows()