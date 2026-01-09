import sys
import cv2 as cv
import os


def main(argv):
    window_name = 'Sobel Demo - Simple Edge Detector'
    scale = 1
    delta = 0
    ddepth = cv.CV_16S

    # Build path to image in input_photo folder
    image_path = os.path.join("input_photo", "lena.jpg")

    src = cv.imread(image_path, cv.IMREAD_COLOR)
    if src is None:
        print("Error opening image:", image_path)
        return -1

    # Reduce noise
    src = cv.GaussianBlur(src, (3, 3), 0)

    # Convert to grayscale
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    # Sobel gradients
    grad_x = cv.Sobel(
        gray, ddepth, 1, 0,
        ksize=3, scale=scale, delta=delta,
        borderType=cv.BORDER_DEFAULT
    )

    grad_y = cv.Sobel(
        gray, ddepth, 0, 1,
        ksize=3, scale=scale, delta=delta,
        borderType=cv.BORDER_DEFAULT
    )

    # Convert gradients to absolute values
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)

    # Combine gradients
    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    cv.imshow(window_name, grad)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
