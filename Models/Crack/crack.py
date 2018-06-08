"""
Faster R-CNN
Train on a road work image dataset

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:
    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco
    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last
    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet
    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>
    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2

from skimage import img_as_float, img_as_ubyte
import tensorflow as tf
import keras.backend as K
from keras.backend.tensorflow_backend import set_session

# Root directory of the project
ROOT_DIR = os.path.abspath("")

# Import Faster RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from frcnn.config import Config
from frcnn import model as modellib, utils

# Path to COCO trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

##################
# Configurations #
##################


class CrackConfig(Config):
    '''
    Configuration for training on the crack dataset.
    Derives from the base Config class and overrides some values.
    '''
    # Give the configuration a recognizable name
    NAME = "crack"

    GPU_COUNT = 1

    # A GPU with 12GB of memory can fit 2 images,
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + crack

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 81

    # Validation steps
    VALIDATION_STEPS = 19

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    TRAIN_ROIS_PER_IMAGE = 25
    MAX_GT_INSTANCES = 15

###########
# Dataset #
###########

class CrackDataset(utils.Dataset):

    def load_crack(self, dataset_dir, subset):
        '''
        Load a subset of the crack dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        '''
        # Add classes. We have only one class to add.
        self.add_class("crack", 1, "crack")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # JSON datasets in the form:
        # path/to/image;
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'x': ...,
        #               'y': ...,
        #               'width': ...,
        #               'height': ...,
        #               'name': 'rect'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            rects = [r['shape_attributes'] for r in a['regions'].values()]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "crack",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=rects)

    def load_mask(self, image_id):
        '''
        Generate instance masks for an image.
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        '''
        # If not a crack dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "crack":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)

        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            #print("Poly no.{}: x={}, y={}, w={}, h={}".format(
            #    i, p['x'], p['y'], p['width'], p['height']))
            # Use the following if your masks are stored as rects
            x = p['x']
            y = p['y']
            # Decrement height and width by one to properly fit box in image
            w = p['width'] - 1
            h = p['height'] - 1

            rr, cc = skimage.draw.polygon(
                [y, y, y + h, y + h],
                [x, x + w, x + w, x])

            # Otherwise if your masks are stored as polygons use the following
            # Bounding boxes will be generated from the polygons
            '''
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            '''
            #print(rr, cc)
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        # Return the path of the image
        info = self.image_info[image_id]
        if info["source"] == "balloon":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    # Training dataset.
    dataset_train = CrackDataset()
    dataset_train.load_crack(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CrackDataset()
    dataset_val.load_crack(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')

def tag_detections(image, rois, class_ids, scores):
    '''
    Tag image detections with corrresponding bboxes, id's, and scores
    Arguments
        image: [H, W, 3] RGB image
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores
    Returns result image.
    '''
    assert len(rois) == len(class_ids) and len(rois) == len(scores), \
        "Length of rois, class_ids, and scores must match"
    
    print("Drawing %d detected bboxes..." % len(scores))
    if (len(scores) > 0):
        font = cv2.FONT_HERSHEY_PLAIN

        # Conver to cv2 format
        img = img_as_ubyte(image)

        for ix in range(len(rois)):
            roi = rois[ix]
            ul_corner = (roi[1], roi[0])
            br_corner = (roi[3], roi[2])

            # Display bbox
            cv2.rectangle(img, ul_corner, br_corner, (0,0,255), thickness=2)

            # Display text (class, score)
            text = "{}-{}".format(class_ids[ix], scores[ix])
            text_size = cv2.getTextSize(text, font, 1, thickness=1)[0]
            ul_corner_text = tuple(map(sum, zip(ul_corner, (0, 15))))
            br_corner_text = tuple(map(sum, zip(ul_corner_text, text_size)))
            cv2.rectangle(img, ul_corner, br_corner_text, (0,0,255), thickness=-1)
            cv2.putText(img, text, ul_corner_text, font, 1, (255,255,255), thickness=1)
        
        # Convert back to skimage format
        image = img_as_float(img)
    
    return image


def detect_and_tag_images(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Tag image with detections
        tagged = tag_detections(image, r['rois'], r['class_ids'], r['scores'])
        # Save output
        file_name = "tagged_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        out_path = os.path.join(os.path.join(ROOT_DIR, "output"), file_name)
        skimage.io.imsave(out_path, tagged)
    elif video_path:
        # Video tagging not implemented
        pass
    print("Saved to ", file_name)


############
# Training #
############

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Faster-CNN to detect cracks.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'tag'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/crack/dataset/",
                        help='Directory of the Crack dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to tag')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to tag')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "tag":
        assert args.image or args.video,\
               "Provide --image to tag"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configure GPU config
    #tf_config = tf.ConfigProto()
    #tf_config.gpu_options.allow_growth = True

    # Configurations
    if args.command == "train":
        config = CrackConfig()
    else:
        class InferenceConfig(CrackConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.FasterRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.FasterRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "frcnn_class_logits", "frcnn_bbox_fc",
            "frcnn_bbox"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate

    if args.command == "train":
        train(model)
    elif args.command == "tag":
        detect_and_tag_images(model, image_path=args.image,
                              video_path=args.video)
    else:
        print("'{}' is not recognized. "
            "Use 'train' or 'tag'".format(args.command))