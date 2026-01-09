import cv2

img = cv2.imread("input_photo/input.jpg", cv2.IMREAD_GRAYSCALE)

laplacian = cv2.Laplacian(img, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

cv2.imshow("Laplacian Edge Detection", laplacian)
cv2.waitKey(0)
cv2.destroyAllWindows()
