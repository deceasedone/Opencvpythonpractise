import cv2
import numpy as np

# Constants for the output image dimensions
WIDTH_IMG = 540
HEIGHT_IMG = 640

# Initialize the camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 100)

def preprocess_image(img):
    """
    Preprocess the input image by converting to grayscale, applying blur,
    edge detection, dilation, and erosion.
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_canny = cv2.Canny(img_blur, 200, 200)
    kernel = np.ones((5, 5))
    img_dilated = cv2.dilate(img_canny, kernel, iterations=2)
    img_thresh = cv2.erode(img_dilated, kernel, iterations=1)
    return img_thresh

def find_contours(img, img_contour):
    """
    Find the largest 4-corner contour in the image for document detection.
    """
    biggest = np.array([])
    max_area = 0
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    if biggest.size != 0:
        cv2.drawContours(img_contour, [biggest], -1, (255, 0, 0), 20)
    return biggest

def reorder_points(points):
    """
    Reorder points to a consistent format: [top-left, top-right, bottom-left, bottom-right].
    """
    points = points.reshape((4, 2))
    reordered = np.zeros((4, 1, 2), dtype=np.int32)
    add = points.sum(axis=1)
    diff = np.diff(points, axis=1)

    reordered[0] = points[np.argmin(add)]
    reordered[3] = points[np.argmax(add)]
    reordered[1] = points[np.argmin(diff)]
    reordered[2] = points[np.argmax(diff)]

    return reordered

def warp_perspective(img, points):
    """
    Perform a perspective warp on the image using the provided points.
    """
    points = reorder_points(points)
    src = np.float32(points)
    dest = np.float32([[0, 0], [WIDTH_IMG, 0], [0, HEIGHT_IMG], [WIDTH_IMG, HEIGHT_IMG]])
    matrix = cv2.getPerspectiveTransform(src, dest)
    img_warped = cv2.warpPerspective(img, matrix, (WIDTH_IMG, HEIGHT_IMG))

    # Crop and resize for clean output
    img_cropped = img_warped[20:img_warped.shape[0] - 20, 20:img_warped.shape[1] - 20]
    img_cropped = cv2.resize(img_cropped, (WIDTH_IMG, HEIGHT_IMG))
    return img_cropped

def stack_images(scale, img_array):
    """
    Stack multiple images into a single window for easier visualization.
    """
    rows = len(img_array)
    cols = len(img_array[0]) if isinstance(img_array[0], list) else 1
    rows_available = isinstance(img_array[0], list)

    width = img_array[0][0].shape[1]
    height = img_array[0][0].shape[0]
    blank_img = np.zeros((height, width, 3), dtype=np.uint8)

    if rows_available:
        for row in range(rows):
            for col in range(cols):
                img_array[row][col] = cv2.resize(
                    img_array[row][col],
                    (width, height),
                    interpolation=cv2.INTER_AREA
                )
                if len(img_array[row][col].shape) == 2:
                    img_array[row][col] = cv2.cvtColor(img_array[row][col], cv2.COLOR_GRAY2BGR)

        hor = [np.hstack(img_row) for img_row in img_array]
        stacked = np.vstack(hor)
    else:
        for i in range(rows):
            img_array[i] = cv2.resize(img_array[i], (width, height), interpolation=cv2.INTER_AREA)
            if len(img_array[i].shape) == 2:
                img_array[i] = cv2.cvtColor(img_array[i], cv2.COLOR_GRAY2BGR)
        stacked = np.hstack(img_array)

    return stacked

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from camera.")
        break

    img = cv2.resize(img, (WIDTH_IMG, HEIGHT_IMG))
    img_contour = img.copy()

    img_thresh = preprocess_image(img)
    biggest_contour = find_contours(img_thresh, img_contour)

    if biggest_contour.size != 0:
        img_warped = warp_perspective(img, biggest_contour)
        images_to_display = [[img_contour, img_warped]]
    else:
        images_to_display = [[img_contour, img]]

    stacked_images = stack_images(0.6, images_to_display)
    cv2.imshow("Workflow", stacked_images)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
