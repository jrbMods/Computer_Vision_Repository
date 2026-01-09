import cv2
import numpy as np

# -----------------------------
# Parameters
# -----------------------------
IMAGE_PATH = "input_photo/input.jpg"  # path to input image
CASCADE_PATH = "haarcascade_frontalface_default.xml"

ANONYMIZATION_METHOD = "pixelate"  
# options: "blur" or "pixelate"

PIXEL_SIZE = 10      # used for pixelation
BLUR_KERNEL = (31, 31)  # used for Gaussian blur

# -----------------------------
# Load image and classifier
# -----------------------------
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise IOError("Cannot load image")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    raise IOError("Cannot load cascade classifier")

# -----------------------------
# Detect faces
# -----------------------------
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

# -----------------------------
# Anonymization functions
# -----------------------------
def anonymize_blur(face_roi):
    return cv2.GaussianBlur(face_roi, BLUR_KERNEL, 0)

def anonymize_pixelate(face_roi, blocks=10):
    (h, w) = face_roi.shape[:2]

    # Resize down
    temp = cv2.resize(
        face_roi,
        (w // blocks, h // blocks),
        interpolation=cv2.INTER_LINEAR
    )

    # Resize up
    pixelated = cv2.resize(
        temp,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    return pixelated

# -----------------------------
# Apply anonymization
# -----------------------------
for (x, y, w, h) in faces:
    face_roi = image[y:y+h, x:x+w]

    if ANONYMIZATION_METHOD == "blur":
        anonymized_face = anonymize_blur(face_roi)
    else:
        anonymized_face = anonymize_pixelate(face_roi, PIXEL_SIZE)

    image[y:y+h, x:x+w] = anonymized_face

# -----------------------------
# Display result
# -----------------------------
cv2.imshow("Face Anonymization", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
