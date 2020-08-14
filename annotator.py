from tqdm import tqdm
import numpy as np
import cv2
from pathlib import Path
import time
import datetime
import argparse


parser = argparse.ArgumentParser(
    description="Enter arguments for image data annotation"
)
parser.add_argument("class_name", type=str, help="annotation class name")
parser.add_argument(
    "-d", "--dataset_path", default="dataset", type=str, help="Path to dataset"
)

args = parser.parse_args()
print("Arguments", args)

# arguments
category = args.class_name

# human readable timestamp for each annotation session
value = datetime.datetime.fromtimestamp(time.time())
timestamp = value.strftime("%Y_%m_%d_%H_%M_%S")

# make dataset directories
data_path = Path(f"{args.dataset_path}/{category}/{timestamp}")
data_path.mkdir(exist_ok=True, parents=True)


cap = cv2.VideoCapture(0)
image_text = ""
counter = 0
counter_bar = tqdm()
image_path = ""
while True:

    # print("Image: ", counter)
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # flip image to remove confusion while collecting data

    if image_text == "o":
        image_name = str(time.time()) + ".png"
        image_path = str(data_path / Path(f"{image_name}"))
        cv2.imwrite(image_path, frame)
        counter += 1
    cv2.putText(
        frame,
        image_text,
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,  # font scale
        (0, 0, 255),  # color
        10,  # thickness
        cv2.LINE_AA,
    )
    # Display the resulting frame
    cv2.imshow("frame", frame)
    k = cv2.waitKey(1)
    if k == ord("q"):
        break
    elif k == 32:  # for spacebar
        if image_text == "o":
            image_text = ""
        else:
            image_text = "o"
    counter_bar.set_description(f"Image: {counter}, Path: {image_path}")

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
