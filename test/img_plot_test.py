import cv2
import skimage.draw
import numpy as np
from skimage import io
from skimage import img_as_float, img_as_ubyte

# Font stuff

font = cv2.FONT_HERSHEY_PLAIN

rois = [(10, 10, 100, 100),
        (150, 150, 350, 350)]
class_ids = [1, 2]
scores = [0.2, 1.0]

colors = {'red':(0,0,255), 'white':(255,255,255)}
class_dict = {0: "BG", 1: "Crack", 2: "Person"}

# Import image as skimage
img = skimage.io.imread("C:/Users/seamus.kirby/Documents/CrackKeras/test/002.jpg")

# Convert to opencv2
img = img_as_ubyte(img)

for ix in range(len(rois)):
    roi = rois[ix]
    ul_corner = (roi[1], roi[0])
    br_corner = (roi[3], roi[2])

    # Display bbox
    cv2.rectangle(img, ul_corner, br_corner, colors['red'], thickness=2)

    # Display text (class, score)
    text = "{}-{}".format(class_dict[class_ids[ix]], scores[ix])
    text_size = cv2.getTextSize(text, font, 1, thickness=1)[0]
    ul_corner_text = tuple(map(sum, zip(ul_corner, (0, 15))))
    br_corner_text = tuple(map(sum, zip(ul_corner_text, text_size)))
    cv2.rectangle(img, ul_corner, br_corner_text, colors['red'], thickness=-1)
    cv2.putText(img, text, ul_corner_text, font, 1, colors['white'], thickness=1)

# Convert back to skimage
img = img_as_float(img)

skimage.io.imshow(img)
skimage.io.show()