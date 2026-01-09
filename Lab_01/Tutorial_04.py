import sys
import cv2 as cv
import os

def main(argv):
    ddepth = cv.CV_16S
    kernel_size = 3
    window_name = "Laplace Demo"
    image_path = os.path.join("input_photo", "lena.jpg")
    src = cv.imread(image_path, cv.IMREAD_COLOR)
    if src is None:
        print("Error opening image:", image_path)
        return -1
    src = cv.GaussianBlur(src, (3, 3), 0)
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)
    dst = cv.Laplacian(src_gray, ddepth, ksize=kernel_size)
    abs_dst = cv.convertScaleAbs(dst)
    cv.imshow(window_name, abs_dst)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return 0

if __name__ == "__main__":
    main(sys.argv[1:])
