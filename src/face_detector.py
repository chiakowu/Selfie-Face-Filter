import numpy as np
import cv2 as cv

# very basic face detector using openCV

def main():
    image_name = '../data/she.jpg'

    face_cascade = cv.CascadeClassifier('../data/haarcascade_frontalface_default.xml')
    img = cv.imread(image_name)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),flags=cv.CASCADE_SCALE_IMAGE)
    for (x,y,w,h) in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    cv.imshow('img',img)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()