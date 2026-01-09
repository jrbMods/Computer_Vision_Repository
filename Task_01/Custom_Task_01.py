import cv2
import numpy as np

# Path to the input image
image_path = "input_photo/input.png"

# Read input image
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found. Check the file path.")
    exit()

# Resize for consistent display (optional)
image = cv2.resize(image, (500, 400))

# Function to show black screen with step text
def show_step(title, wait_time=1000):
    screen = np.zeros((400, 500, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    thickness = 2

    text_size = cv2.getTextSize(title, font, scale, thickness)[0]
    text_x = (screen.shape[1] - text_size[0]) // 2
    text_y = (screen.shape[0] + text_size[1]) // 2

    cv2.putText(screen, title, (text_x, text_y),
                font, scale, (255, 255, 255), thickness)

    cv2.imshow("Tutorial Progress", screen)
    cv2.waitKey(wait_time)
    cv2.destroyAllWindows()

# ---- Step 0: Original Image ----
show_step("STEP 0: ORIGINAL IMAGE")
cv2.imshow("Original Image", image)
cv2.waitKey(1000)
cv2.destroyAllWindows()

# ---- Step 1: Grayscale ----
show_step("STEP 1: GRAYSCALE CONVERSION")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale Image", gray)
cv2.waitKey(1000)
cv2.destroyAllWindows()

# ---- Step 2: Gaussian Blur ----
show_step("STEP 2: GAUSSIAN FILTERING")
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imshow("Blurred Image", blurred)
cv2.waitKey(1000)
cv2.destroyAllWindows()

# ---- Step 3: Edge Detection ----
show_step("STEP 3: EDGE DETECTION (CANNY)")
edges = cv2.Canny(blurred, 50, 150)
cv2.imshow("Edge Image", edges)
cv2.waitKey(1000)
cv2.destroyAllWindows()

# ---- Step 4: Thresholding ----
show_step("STEP 4: BINARY THRESHOLDING")
_, binary = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
cv2.imshow("Binary Image", binary)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save final output
cv2.imwrite("saved_photo/binary_output.jpg", binary)