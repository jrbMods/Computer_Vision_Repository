import cv2
import numpy as np
import math

# ---------- Load Image ----------
image = cv2.imread("input_photo/input.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ---------- Preprocessing ----------
# Blur to remove noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Threshold (binary image)
_, thresh = cv2.threshold(
    blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# Morphological Opening (remove small noise)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# ---------- Find Contours ----------
contours, _ = cv2.findContours(
    opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

print(f"Total Objects Detected: {len(contours)}")

# ---------- Analyze Each Contour ----------
for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    if area < 50:  # Ignore very small objects
        continue

    # ---- Moments ----
    M = cv2.moments(cnt)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # ---- Bounding Rectangle ----
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h
    extent = area / (w * h)

    # ---- Convex Hull ----
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area != 0 else 0

    # ---- Equivalent Diameter ----
    equi_diameter = np.sqrt(4 * area / np.pi)

    # ---- Orientation ----
    angle = 0
    if len(cnt) >= 5:
        (_, _), (_, _), angle = cv2.fitEllipse(cnt)

    # ---- Mean Intensity ----
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    mean_intensity = cv2.mean(gray, mask=mask)[0]

    # ---- Draw Results ----
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)

    # ---- Print Object Features ----
    print(f"\nObject {i + 1}")
    print(f" Center: ({cx}, {cy})")
    print(f" Aspect Ratio: {aspect_ratio:.2f}")
    print(f" Extent: {extent:.2f}")
    print(f" Solidity: {solidity:.2f}")
    print(f" Equivalent Diameter: {equi_diameter:.2f}")
    print(f" Orientation: {angle:.2f} degrees")
    print(f" Mean Intensity: {mean_intensity:.2f}")

# ---------- Show Output ----------
cv2.imshow("Detected Objects", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
