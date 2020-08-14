from tqdm import tqdm
import numpy as np
import cv2
from pathlib import Path
import time
import datetime
import argparse


cap = cv2.VideoCapture(0)
image_text = ""
counter = 0
# counter_bar = tqdm()
image_path = ""
while True:

    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # flip image to remove confusion while collecting data

    # do inference

    # do inference annotation

    # Display the resulting frame
    cv2.imshow("frame", frame)
    k = cv2.waitKey(1)
    if k == ord("q"):
        break
    # counter_bar.set_description(f"Image: {counter}, Path: {image_path}")

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
