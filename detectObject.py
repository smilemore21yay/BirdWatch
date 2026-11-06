import cv2 as cv
import numpy as np

print(hasattr(cv, 'CascadeClassifier'))


"""
Captures the video footage from the laptop 

"""
cap = cv.VideoCapture(0) #Getting the laptops video camera footage

if not cap.isOpened():
    print("Error: Could not open the camera")
else:
    print("Camera opened successfuly")


#Where we start to detect the bird
bird_cascade = cv.CascadeClassifier('bird-cascade.xml')

while True:
    #Reading the first camera frame
    ret, frame = cap.read()

    found_img = bird_cascade.detectMultiScale(frame, minSize=(20, 20))

    for (x,y,w,h) in found_img:
        cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 5)

    cv.imshow('Frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):  #Waits for a key press to close the window
        break


cap.release()
cv.destroyAllWindows()   #Closes the window
