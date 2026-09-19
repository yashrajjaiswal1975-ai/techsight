import cv2
import time

cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

print("Camera opened:", cap.isOpened())

start = time.time()
count = 0

while time.time() - start < 20:

    ret, frame = cap.read()

    if not ret:
        print("FAILED TO READ FRAME")
        break

    count += 1

    if count % 10 == 0:
        print("Frames received:", count)

cap.release()

print("Test finished.")
print("Total frames:", count)
