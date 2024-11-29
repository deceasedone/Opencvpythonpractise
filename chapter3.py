import cv2
img = cv2.imread("Resources/lambo.png")
cv2.imshow("Image", img)
print(img.shape)
imgResize=cv2.resize(img,(1000,200))
cv2.imshow("resized image", imgResize)
imgCropped = img[0:200,200:500]
print(imgResize.shape)
cv2.imshow("cropped image", imgCropped)
print(imgCropped.shape)
cv2.waitKey(0)