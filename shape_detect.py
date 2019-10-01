#!/usr/bin/env python
import numpy as np
import cv2

cap = cv2.VideoCapture(0)


def circleDetect(img, cnt, approx):
    x = approx.ravel()[0]
    y = approx.ravel()[1]
    cv2.putText(img, 'Circle', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
    cv2.drawContours(img, [cnt], 0, (0, 255, 255), -1)


def squareDetect(img, cnt, approx):
    x = approx.ravel()[0]
    y = approx.ravel()[1]
    cv2.putText(img, 'Square', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
    cv2.drawContours(img, [cnt], 0, (0, 0, 255), -1)


while True:
    ret, frame = cap.read()

    # Turn image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Remove gaussian noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(blur, kernel)

    # Extract contours from frame
    contours, _ = cv2.findContours(
        mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # For each large contour detected
    for cnt in contours:

        # Get the area of the contour
        if cv2.contourArea(cnt) < 400:
            continue

        # Approximate the number of sizes to the shape
        approx = cv2.approxPolyDP(
            cnt, 0.01 * cv2.arcLength(cnt, True), True)

        # if 4 sided shape
        if len(approx) == 4:
            squareDetect(frame, cnt, approx)
        # if greater than 15, assume circle
        elif len(approx) > 15:
            circleDetect(frame, cnt, approx)

    cv2.imshow('frame', frame)

    cv2.waitKey(3)

cap.release()
cv2.destroyAllWindows()
