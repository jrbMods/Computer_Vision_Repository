import cv2
import numpy as np

# ----------- CONFIG -----------
USE_TEMPLATE_MATCHING = True
USE_CONTOUR_MATCHING = True
# ------------------------------

image = cv2.imread("input_photo/scene.jpg")
template = cv2.imread("input_photo/template.jpg")

gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_tpl = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# ==============================
# TEMPLATE MATCHING
# ==============================
if USE_TEMPLATE_MATCHING:
    result = cv2.matchTemplate(gray_img, gray_tpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    h, w = gray_tpl.shape
    cv2.rectangle(
        image,
        max_loc,
        (max_loc[0] + w, max_loc[1] + h),
        (255, 0, 0),
        2
    )

    print(f"Template match score: {max_val:.2f}")

# ==============================
# CONTOUR SHAPE MATCHING
# ==============================
if USE_CONTOUR_MATCHING:
    _, thresh_img = cv2.threshold(gray_img, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, thresh_tpl = cv2.threshold(gray_tpl, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours_img, _ = cv2.findContours(thresh_img,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
    contours_tpl, _ = cv2.findContours(thresh_tpl,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

    template_contour = max(contours_tpl, key=cv2.contourArea)

    for cnt in contours_img:
        score = cv2.matchShapes(template_contour, cnt,
                                cv2.CONTOURS_MATCH_I1, 0)
        if score < 0.1:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image, (x, y),
                          (x + w, y + h), (0, 255, 0), 2)
            print(f"Shape match score: {score:.4f}")

# ----------- SHOW RESULT -----------
cv2.imshow("Object Detection Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
