import cv2
import numpy as np

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 100)

# Define colors (HSV ranges) and corresponding BGR values
myColors = [
    [5, 107, 0, 19, 255, 255],  # Orange
    [133, 56, 0, 159, 156, 255],  # Purple
    [57, 76, 0, 100, 255, 255],  # Green
    [90, 48, 0, 118, 255, 255]  # Blue
]

myColorValues = [
    [51, 153, 255],  # Orange in BGR
    [255, 0, 255],  # Purple in BGR
    [0, 255, 0],  # Green in BGR
    [255, 0, 0]  # Blue in BGR
]

# Check if colors and values are consistent
if len(myColors) != len(myColorValues):
    raise ValueError("myColors and myColorValues must have the same length.")

# Store drawn points as [x, y, colorId]
myPoints = []


def findColor(img, myColors, myColorValues):
    """Find the specified colors in the image and return new points."""
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    newPoints = []
    for count, color in enumerate(myColors):
        lower = np.array(color[:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(imgHSV, lower, upper)
        x, y = getContours(mask)
        if x != 0 and y != 0:
            cv2.circle(imgResult, (x, y), 15, myColorValues[count], cv2.FILLED)
            newPoints.append([x, y, count])
        # Debug: Uncomment to visualize individual masks
        # cv2.imshow(f"Mask {count}", mask)
    return newPoints


def getContours(img):
    """Extract contours and find the center of the largest contour."""
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x, y, w, h = 0, 0, 0, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            # Draw the contour (optional, for debugging)
            # cv2.drawContours(imgResult, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
    return x + w // 2, y


def drawOnCanvas(myPoints, myColorValues):
    """Draw the stored points on the canvas."""
    for point in myPoints:
        cv2.circle(imgResult, (point[0], point[1]), 10, myColorValues[point[2]], cv2.FILLED)


while True:
    success, img = cap.read()
    imgResult = img.copy()

    # Find new points and add them to the existing list
    newPoints = findColor(img, myColors, myColorValues)
    if newPoints:
        myPoints.extend(newPoints)

    # Draw all points on the canvas
    if myPoints:
        drawOnCanvas(myPoints, myColorValues)

    cv2.imshow("Result", imgResult)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
