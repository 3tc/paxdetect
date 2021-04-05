import numpy as np
import cv2
from datetime import datetime
from os import environ

env=dict(environ)
video_url =env["CAMERA1_URL"] 

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# sub stream 640x480 H264 - MPEG-4AVC (part 10)(h264)
cap = cv2.VideoCapture(video_url)

count = 0

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # resizing for faster detection
    # frame = cv2.resize(frame, (640, 480))
    # using a greyscale picture, also for faster detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # detect people in the image
    # returns the bounding boxes for the detected objects
    boxes, weights = hog.detectMultiScale(gray, winStride=(2,2), padding=(10, 10), scale=1.02)

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    if boxes.size > 0 and weights[0] > .9:
        count += 1
        ts = int(datetime.timestamp(datetime.now()))
        fileprefix="{}-{}".format(ts, count)

        # write image without detection regions to assist with training - S suffix
        print("Detection {} {} - weights {}, rect {}".format(fileprefix, count, weights, boxes))
        cv2.imwrite("output/{}S.jpg".format(fileprefix), frame)

        for (xA, yA, xB, yB) in boxes:
            # display the detected boxes in the colour picture
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
    
        # write image with detection regions - D suffix
        cv2.imwrite("output/{}D.jpg".format(fileprefix), frame)

# When everything done, release the capture
cap.release()
# and release the output
out.release()


