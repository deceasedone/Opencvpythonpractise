import cv2
'''
print("Package imported")

img=cv2.imread("Resources/lena.png")
cv2.imshow("Output",img)
cv2.waitKey(0)
'''
'''
cap = cv2.VideoCapture("Resources/test_video.mp4")
while True:
    success, img = cap.read()
    cv2.imshow ("Video", img)
    if cv2.waitKey(3) & 0xFF == ord('q'):
        break
'''
frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 100)
while True:
    success, img = cap.read()
    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break