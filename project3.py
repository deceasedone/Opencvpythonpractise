import cv2
import os

# Constants
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MIN_AREA = 200
COLOR = (255, 0, 255)
CASCADE_PATH = "Resources/haarcascades/haarcascade_russian_plate_number.xml"
VIDEO_PATH = "Resources/video12.mp4"
SAVE_DIR = "Resources/Scanned"

# Load Haar Cascade
if not os.path.exists(CASCADE_PATH):
    raise FileNotFoundError(f"Haar Cascade file not found at: {CASCADE_PATH}")
plate_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# Create save directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

# Initialize Video Capture
cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)

count = 0


def detect_and_draw_plates(frame, cascade, min_area):
    """
    Detects number plates in the given frame and draws rectangles around them.
    """
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plates = cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=10)

    rois = []  # Store regions of interest (ROI)
    for (x, y, w, h) in plates:
        area = w * h
        if area > min_area:
            # Draw rectangle around detected plate
            cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR, 2)
            cv2.putText(frame, "Number Plate", (x, y - 5),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, COLOR, 2)
            # Store the cropped region of interest
            roi = frame[y:y + h, x:x + w]
            rois.append(roi)
    return frame, rois


def save_roi(roi, count, save_dir):
    """
    Saves the detected ROI to the specified directory with a unique name.
    """
    save_path = os.path.join(save_dir, f"NoPlate_{count}.jpg")
    cv2.imwrite(save_path, roi)
    print(f"Saved: {save_path}")


def main():
    global count
    while True:
        success, frame = cap.read()
        if not success:
            print("Video has ended or failed to load.")
            break

        # Detect number plates
        annotated_frame, rois = detect_and_draw_plates(frame, plate_cascade, MIN_AREA)

        # Display the main video feed with annotations
        cv2.imshow("Result", annotated_frame)

        # Display each detected ROI in a separate window
        for i, roi in enumerate(rois):
            cv2.imshow(f"ROI {i}", roi)

        # Save ROI if 's' key is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and rois:
            for roi in rois:
                save_roi(roi, count, SAVE_DIR)
                count += 1
                # Provide visual feedback after saving
                cv2.rectangle(annotated_frame, (0, 200), (FRAME_WIDTH, 300), (0, 255, 0), cv2.FILLED)
                cv2.putText(annotated_frame, "Scan Saved", (150, 265),
                            cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 2)
                cv2.imshow("Result", annotated_frame)
                cv2.waitKey(500)

        # Exit the loop if 'q' key is pressed
        if key == ord('q'):
            print("Exiting the application.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
