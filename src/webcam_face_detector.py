import numpy as np
import cv2 as cv

def main():
    face_classifier = cv.CascadeClassifier('../data/haarcascade_frontalface_default.xml')
    capture = cv.VideoCapture(0)

    while True:
        ret, frame = capture.read()

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                flags=cv.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x+w, x+h), (0,255,0), 2)

        cv.imshow('Video', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()