import cv2

img = cv2.imread("input_photo/input.jpg", cv2.IMREAD_GRAYSCALE)

sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)

sobel = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

cv2.imshow("Sobel Edge Detection", sobel)
cv2.waitKey(0)
cv2.destroyAllWindows()
