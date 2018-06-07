## Implementation of a Faster R-CNN network using
## a Feature pyramid network for object detection

## Network design:
##
## Image input (Pad/resize to normalize image inputs)
## 
## FPN (Built with ResNet50 backbone, generates multiple
##      feature maps of different sizes with lateral residual connections)
## Conv1 -> M1
##   ▼      ▲
## Conv2 -> M2
##   ▼      ▲
## Conv3 -> M3
##
## RPN (Takes in all feature maps and generates ROIs,
##      selects the feature map with the best scale
##      based on size of ROI to generate ROI patches)
##
## ROI pooling (Normalizes ROI patches to uniform size)
##
## FC Layers (Last layers branch into a classifier and a
##            regressor, to simultaneously classify ROIs
##            and output a bounding box)

from __future__ import print_function

import math
import os
import random
import datetime
import re
import logging
import multiprocessing
import warnings

import keras
import keras.backend as K
import keras.engine as KE
import numpy as np
import tensorflow as tf
from keras import layers
from keras.applications.imagenet_utils import (_obtain_input_shape,
                                               decode_predictions,
                                               preprocess_input)
from keras.engine.topology import get_source_inputs
from keras.layers import (Activation, AveragePooling2D, BatchNormalization,
                          Conv2D, Dense, Flatten, GlobalAveragePooling2D,
                          GlobalMaxPooling2D, Input, Lambda, MaxPooling2D,
                          Reshape, TimeDistributed, UpSampling2D,
                          ZeroPadding2D)
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.regularizers import l2
from keras.utils import layer_utils
from keras.utils.data_utils import get_file

from frcnn import utils

#########
# TODO: #
#########
'''
Allow for other models (alexnet for example)


TensorFlow functions to replace 
tf.nn.top_k()
tf.where()
tf.boolean_mask()
tf.image.non_maximum_suppression()
'''

#####################
# Utility Functions #
#####################

def log(text, array=None):
    '''
    Prints a text message, and if a Numpy array is provided print it's
    shape, min, and max values.
    '''
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20} min: {:10.5f} max: {:10.5f} {}".format(
            str(array.shape),
            array.min() if array.size else "",
            array.max() if array.size else "",
            array.dtype))
    print(text)

class BatchNorm(BatchNormalization):
    '''
    Extends the Keras BatchNormalization class to allow a central place to
    make changes if necessary
    '''
    def call(self, inputs, training=None):
            '''
            Note about training values:
                None: Train BN layers (default)
                False: Freeze BN layers. Good when batch size is small
                True: (Don't use). Set layer in training mode even when
                      inferencing
            '''
            return super().call(inputs, training=training)

def compute_backbone_shapes(config, image_shape):
    '''
    Computes the width and hiegh of each stage of the backbone network
    Returns
        [N, (height, width)] where N is the number of stages
    '''
    # Currently only works for ResNet
    assert config.BACKBONE in ["resnet50"] #["resnet50", "resnet101"]
    return np.array(
        [[int(math.ceil(image_shape[0] / stride)),
        int(math.ceil(image_shape[1] / stride))]
        for stride in config.BACKBONE_STRIDES])


#############################
# Backbone Model (Resnet50) #
#############################

## Note: Found resnet architecture code at
## https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

def identity_block(input_tensor, kernel_size, filters, stage, block,
                    use_bias=True, train_bn=True):
    '''
    Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean, Whether or not to use bias in conv layers
        train_bn: Boolean, train or freeze batc normalization layers
    Returns
        Output tensor for the block
    '''

    filters1, filters2, filters3 = filters
    if K.image_data_format == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a',
                use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', 
                name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    x = layers.add([x, input_tensor])
    x = Activation('relu', name='res' + str(stage) + block + '_out')(x)
    
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2),
                use_bias=True, train_bn=True):
    '''
    Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean, Whether or not to use bias in conv layers
        train_bn: Boolean, train or freeze batc normalization layers
    Returns
        Output tensor for the block.
    '''

    filters1, filters2, filters3 = filters
    if K.image_data_format == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a',
                use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Conv2D(filters2, (kernel_size, kernel_size), padding='same', 
                name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1',
                        use_bias=use_bias)(input_tensor)
    shortcut = BatchNorm(name=bn_name_base + '1')(shortcut, training=train_bn)

    x = layers.add([x, shortcut])
    x = Activation('relu', name='res' + str(stage) + block + '_out')(x)
    
    return x

def resnet_graph(input_image, stage5=False, train_bn=True):
    '''
    Build a ResNet graph with ResNet50 architecture
    Arguments
        input_image: tensor for image input
        train_bn: Boolean, whether to train batch normalization layers
    '''

    # Stage 1
    x = ZeroPadding2D((3, 3))(input_image)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = BatchNorm(name='bn_conv1')(x, training=train_bn)
    x = Activation('relu')(x)
    C1 = x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)

    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)

    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
    # ResNet50 uses an addiitonal 5 identity blocks in stage 4
    for i in range(5):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98+i), train_bn=train_bn)
    C4 = x

    # Stage 5
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)
    else:
        C5 = None

    return [C1, C2, C3, C4, C5]

##################
# Proposal Layer #
##################

def apply_box_deltas_graph(boxes, deltas):
    '''
    Applies the given deltas to the given boxes
    Arguments
        boxes: [N, (y1, x1, y2, x2)] boxes to update
        deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
    Returns
        result: [N, (y1, x1, y2, x2)] updated boxes
    '''

    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width

    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= K.exp(deltas[:, 2])
    width *= K.exp(deltas[:, 3])

    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
    return result

def clip_boxes_graph(boxes, window):
    '''
    Arguments
        boxes: [N, (y1, x1, y2, x2)]
        window: [y1, x1, y2, x2]
    '''

    # Split
    '''
    wy1 = window[0]
    wx1 = window[1]
    wy2 = window[2]
    wx2 = window[3]
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    '''
    
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)

    # Clip
    y1 = K.maximum(K.minimum(y1, wy2), wy1)
    x1 = K.maximum(K.minimum(x1, wx2), wx1)
    y2 = K.maximum(K.minimum(y2, wy2), wy1)
    x2 = K.maximum(K.minimum(x2, wx2), wx1)
    
    clipped = K.concatenate([y1, x1, y2, x2], axis=1)
    clipped.set_shape((clipped.shape[0], 4))
    return clipped

class ProposalLayer(KE.Layer):
    '''
    Receives anchor scores and selets a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores 
    and non-max suppression to remove overlaps. It also applies bounding
    and box refinement deltas to anchors
    
    Inputs:
        rpn_probs: [batch, anchors, (bg_prob, fg_prob)]
        rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
        anchors: [batch, (y1, x1, y2, x2)] in normalized coords
    Returns
        [batch, rois, (y1, x1, y2, x2)] proposals in normalized coords 
    '''

    def __init__(self, proposal_count, nms_threshold, config=None, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.config = config
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold

    def call(self, inputs):
        scores = inputs[0][:, :, 1]
        deltas = inputs[1]
        deltas = deltas * np.reshape(self.config.RPN_BBOX_STD_DEV, [1, 1, 4])
        anchors = inputs[2]

        #assert K.shape(scores)[0] == K.shape(anchors)[0], "Error! Scores and anchors don't match"
        #print("Scores: " + str(K.shape(scores)[1]))
        #print("Deltas: " + str(K.shape(deltas)[1]))
        #print("Anchors: " + str(K.shape(anchors)[1]))

        # TODO: Convert following code block to keras
        # Improve performance by trimming to top anchors by score
        # and doing the rest on the smaller subset.
        
        pre_nms_limit = K.minimum(6000, K.shape(anchors)[1])
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True,
                         name="top_anchors").indices
        scores = utils.batch_slice([scores, ix], lambda x, y: K.gather(x, y),
                                   self.config.IMAGES_PER_GPU)
        deltas = utils.batch_slice([deltas, ix], lambda x, y: K.gather(x, y),
                                   self.config.IMAGES_PER_GPU)
        pre_nms_anchors = utils.batch_slice([anchors, ix], lambda a, x: K.gather(a, x),
                                    self.config.IMAGES_PER_GPU,
                                    names=["pre_nms_anchors"])
        

        # Refine anchors by applying deltas
        boxes = utils.batch_slice([pre_nms_anchors, deltas],
                                  lambda x, y: apply_box_deltas_graph(x, y),
                                  self.config.IMAGES_PER_GPU,
                                  names=["refined_anchors"])

        # Clip to image boundaries
        window = np.array([0, 0, 1, 1], dtype=np.float32)
        boxes = utils.batch_slice(boxes,
                                  lambda x: clip_boxes_graph(x, window),
                                  self.config.IMAGES_PER_GPU,
                                  names=["refined_anchors_clipped"])

        # TODO: Implement Non Max Suppression
        # Non Max Suppression
        def nms(boxes, scores):
            indices = tf.image.non_max_suppression(
                boxes, scores, self.proposal_count,
                self.nms_threshold, name="rpn_non_max_suppression")
            # indices should be a 1D array of indices for bboxes not
            # removed by NMS
            proposals = K.gather(boxes, indices)
            # Pad if needed (K.pad does not exist, should pad proposals)
            padding = K.maximum(self.proposal_count - K.shape(proposals)[0], 0)
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            return proposals
        
        proposals = utils.batch_slice([boxes, scores], nms, 
                                      self.config.IMAGES_PER_GPU)
        return proposals

    def compute_output_shape(self, input_shape):
        return (None, self.proposal_count, 4)


##################
# ROIAlign Layer #
##################

def log2_graph(x):
    # Returns log2 of x
    return K.log(x) / K.log(2.0)

class PyramidROIAlign(KE.Layer):
    '''
    Implements ROI pooling on multiple levels of the feature pyramid

    Params
        pool_shape: [height, width] of the output pooled regions. Usually [7, 7]
    Inputs
        boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords.
               Possibly padded with zeros in not enough boxes to fill array
        image_meta: [batch, (meta_data)] Image details, see compose_image_meta()
        feature_maps: List of feature maps from different levels of the pyramid.
                      Each is [batch, height, width, channels]
    Output:
        Pooled regions in the shape: [batch, num_boxes, height, width, channels].
        The width and height are those specific in the pool_shape in the layer
        constructor.
    '''

    def __init__(self, pool_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)

    def call(self, inputs):
        boxes = inputs[0]
        image_meta = inputs[1]
        feature_maps = inputs[2:]

        # Assign each ROI to a level in the pyramid based on the ROI area
        '''
        y1 = boxes[:, :, 2][0]
        x1 = boxes[:, :, 2][1]
        y2 = boxes[:, :, 2][2]
        x2 = boxes[:, :, 2][3]
        '''
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1
        # Use shape of first image. Images in a batch must have the same size
        image_shape = parse_image_meta_graph(image_meta)['image_shape'][0]
        # Apply equation 1 in Feature Pyramid Network paper
        # See https://arxiv.org/abs/1612.03144 for more info
        image_area = K.cast(image_shape[0] * image_shape[1], "float32")
        roi_level = log2_graph(K.sqrt(h * w) / (224. / K.sqrt(image_area)))
        roi_level = K.minimum(5, K.maximum(2, 4 + tf.cast(K.round(roi_level), tf.int32)))
        roi_level = K.squeeze(roi_level, 2)

        # Loop through levels and apply ROI pooling to each. P2 -> P5
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(K.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)

            # Box indices for crop_and_resize
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # Keep track of which box mapped to which level
            box_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            level_boxes = K.stop_gradient(level_boxes)
            box_indices = K.stop_gradient(box_indices)

            # Crop and resize
            pooled.append(tf.image.crop_and_resize(feature_maps[i],
                level_boxes, box_indices, self.pool_shape, method="bilinear"))
        
        # Pack pooled features into a single tensor
        pooled = K.concatenate(pooled, axis=0)

        # Pack box_to_level mapping into one array and add another
        # column representing the order of pooled boxes
        box_to_level = K.concatenate(box_to_level, axis=0)
        box_range = K.expand_dims(K.arange(K.shape(box_to_level)[0]), 1)
        box_to_level = K.concatenate([tf.cast(box_to_level, tf.int32), box_range], axis=1)

        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch and then by index
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=K.shape(box_to_level)[0]).indices[::-1]
        ix = K.gather(box_to_level[:, 2], ix)
        pooled = K.gather(pooled, ix)
        
        # Re add the batch dimension
        pooled = K.expand_dims(pooled, 0)
        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1],)

##########################
# Detection Target Layer #
##########################

def overlaps_graph(boxes1, boxes2):
    '''
    Computes IoU overlaps (area of overlap divided by area of union)
    between two sets of boxes
    Arguments
        boxes1, boxes2: [N, (y1, x1, y2, x2)]
    Returns
        overlaps: IoU overlaps
    '''

    # Tile boxes2 and repeat boxes1, so that we can compare
    # every boxes1 against every boxes2 without looping
    # TODO: Replace b1 method with K.repeat
    b1 = K.reshape(K.tile(K.expand_dims(boxes1, 1),
                   [1, 1, K.shape(boxes2)[0]]), [-1, 4])   
    b2 = K.tile(boxes2, [K.shape(boxes1)[0], 1])

    # Compute intersections
    '''
    b1_y1 = b1[:, 1][0]
    b1_x1 = b1[:, 1][1]
    b1_y2 = b1[:, 1][2]
    b1_x2 = b1[:, 1][3]
    b2_y1 = b2[:, 1][0]
    b2_x1 = b2[:, 1][1]
    b2_y2 = b2[:, 1][2]
    b2_x2 = b2[:, 1][3]
    '''
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    y1 = K.maximum(b1_y1, b2_y1)
    x1 = K.maximum(b1_x1, b2_x1)
    y2 = K.minimum(b1_y2, b2_y2)
    x2 = K.minimum(b1_x2, b2_x2)
    intersection = K.maximum(x2 - x1, 0) * K.maximum(y2 - y1, 0)

    # Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection

    # Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    return K.reshape(iou, [K.shape(boxes1)[0], K.shape(boxes2)[0]])

def detection_targets_graph(proposals, gt_class_ids, gt_boxes, config):
    '''
    Generates detection targets for one image. Subsamples proposals
    and generates class IDs, and bounding box deltas for each
    
    Inputs
        proposals: [N, (y1, x1, y2, x2)] in normalized coords. Might be
                   zero padded if there were not enough proposals
        gt_class_ids: [MAX_GT_INSTANCES] int class IDs
        gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized coords
    Returns
        rois: [TRAIN_ROIS_PER_IMAGE, (y1 ,x1, y2, x2)] in normalized coords
        class_ids: [TRAIN_ROIS_PER_IMAGE] int class IDs, zero padded
        deltas = [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (dy, dx, log(dg), log(dy))]
                 Class-specific bounding box refinements
    
    Note: Returned arrays might be zero-padded if not enough target ROIs
    '''

    # Remove zero padding
    # TODO: trim_zeros_graph() continued, no boolean mask method in keras
    # use K.gather or something (maybe K.lambda???)
    proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
    gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
    gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros, name="trim_gt_class_ids")

    # Handle COCO (Common Objects in Common Context) crowds
    # A crowd box in COCO is a bounding box around several instances.
    # Exclude them from training, crowd boxes are given negative class IDs
    crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
    non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
    crowd_boxes = K.gather(gt_boxes, crowd_ix)
    gt_class_ids = K.gather(gt_class_ids, non_crowd_ix)
    gt_boxes = K.gather(gt_boxes, non_crowd_ix)

    # Compute overlaps matrix [proposals, gt_boxes]
    overlaps = overlaps_graph(proposals, gt_boxes)

    # Compute overlaps with crowd boxes
    crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
    crowd_iou_max = K.max(crowd_overlaps, axis=1)
    no_crowd_bool = (crowd_iou_max < 0.001)

    # Determine positive and negative ROIs
    roi_iou_max = K.max(overlaps, axis=1)
    # Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = (roi_iou_max >= 0.5)
    positive_indices = tf.where(positive_roi_bool)[:, 0]
    # Negative ROIs are those < 0.5 IoU with every GT box
    negative_indices = tf.where(tf.logical_and(roi_iou_max < 0.5, no_crowd_bool))[:, 0]

    # Subsample ROIs. Aim for 33% positive
    # Positive ROIs
    positive_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
    positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
    positive_count = K.shape(positive_indices)[0]
    # Negative ROIs. Add enough to mantain positive:negative ratio
    r = 1. / config.ROI_POSITIVE_RATIO
    negative_count = K.cast(r * K.cast(positive_count, "float32"), "int32") - positive_count
    negative_indices = tf.random_shuffle(negative_indices)[:negative_count]
    # Gather selected ROIs
    positive_rois = K.gather(proposals, positive_indices)
    negative_rois = K.gather(proposals, negative_indices)

    # Assign positive ROIs to GT boxes
    positive_overlaps = K.gather(overlaps, positive_indices)
    roi_gt_box_assignment = K.argmax(positive_overlaps, axis=1)
    roi_gt_boxes = K.gather(gt_boxes, roi_gt_box_assignment)
    roi_gt_class_ids = K.gather(gt_class_ids, roi_gt_box_assignment)

    # Compute bounding box refinement for positive ROIs
    deltas = utils.box_refinement_graph(positive_rois, roi_gt_boxes)
    deltas /= config.BBOX_STD_DEV

    # Append negative ROIs and pad bounding box deltas that are not
    # used for negative ROIs with zeros
    rois = K.concatenate([positive_rois, negative_rois], axis=0)
    N = K.shape(negative_rois)[0]
    P = K.maximum(config.TRAIN_ROIS_PER_IMAGE - K.shape(rois)[0], 0)
    rois = tf.pad(rois, [(0, P), (0, 0)])
    rois_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
    deltas = tf.pad(deltas, [(0, N + P), (0, 0)])

    return rois, roi_gt_class_ids, deltas

class DetectionTargetLayer(KE.Layer):
    '''
    Subsamples proposals and generates target box refinement, and class_ids for each 
    Inputs
        proposals: [batch, N, (y1, x1, y2, x2)] in normalized coords
        gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
        gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized coords
    Returns
        rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coords
        target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE] Integer class IDs
        target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (dy, dx, log(dh), log(dw), class_id)]
                       Class specific bounding box refinements
    Note: Input and returned arrays might be zero padded if not enough target ROIs    
    '''

    def __init__(self, config, **kwargs):
        super(DetectionTargetLayer, self).__init__(**kwargs)
        self.config = config
    
    def call(self, inputs):
        proposals = inputs[0]
        gt_class_ids = inputs[1]
        gt_boxes = inputs[2]
    
        # Slice the batch and run a graph for each slice
        names = ["rois", "target_class_ids", "target_bbox"]
        outputs = utils.batch_slice(
            [proposals, gt_class_ids, gt_boxes],
            lambda x, y, z: detection_targets_graph(
                x, y, z, self.config),
            self.config.IMAGES_PER_GPU, names=names)
        return outputs

    def compute_output_shape(self, input_shape):
        return [
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4), # rois
            (None, 1), # class_ids
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # deltas
        ]

###################
# Detection Layer #
###################

def refine_detections_graph(rois, probs, deltas, window, config):
    '''
    Refine classified proposals and filter overlaps and return final detections
    Inputs
        rois: [N, (y1, x1, y2, x2)] in normalized coords
        probs: [N, num_classes] class probabilities
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))] class specific
                bounding box refinements
        window: (y1, x1, y2, x2) in image coordinates, the portion of the image
                that contains the image excluding te padding
    Returns
        detections: [N, (y1, x1, y2, x2, class_id, score)] in normalized coords
    '''
    # Class IDs per ROI
    class_ids = tf.argmax(probs, axis=1, output_type="int32")
    # Class probability of the top class of each ROI
    indices = K.stack([K.arange(probs.shape[0]), class_ids], axis=1)
    class_scores = tf.gather_nd(probs, indices)
    # Class specific bounding box deltas
    deltas_specific = tf.gather_nd(deltas, indices)
    # Apply bounding box deltas
    refined_rois = apply_box_deltas_graph(
        rois, deltas_specific * config.BBOX_STD_DEV)
    # Clip boxes to image window
    refined_rois = clip_boxes_graph(refined_rois, window)

    # TODO: Filter out boxes with zero area
    
    # Filter out background boxes
    keep = tf.where(class_ids > 0)[:, 0]
    # Filter out low confidence boxes
    if config.DETECTION_MIN_CONFIDENCE:
        conf_keep = tf.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[:, 0]
        keep = tf.sets.set_intersection(K.expand_dims(keep, 0),
                                        K.expand_dims(conf_keep, 0))
        keep = K.to_dense(keep)[0]
    
    # Apply per-class NMS
    pre_nms_class_ids = K.gather(class_ids, keep)
    pre_nms_scores = K.gather(class_scores, keep)
    pre_nms_rois = K.gather(refined_rois, keep)
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    def nms_keep_map(class_id):
        '''
        Apply Non-Maximum Suppression on ROIs of the given class
        '''
        # Indices of ROIs of the given class
        ixs = tf.where(K.equal(pre_nms_class_ids, class_id))[:, 0]
        # Apply NMS
        class_keep = tf.image.non_max_suppression(
            K.gather(pre_nms_rois, ixs),
            K.gather(pre_nms_scores, ixs),
            max_output_size=config.DETECTION_MAX_INSTANCES,
            iou_threshold=config.DETECTION_NMS_THRESHOLD)
        # Map indices
        class_keep = K.gather(keep, K.gather(ixs, class_keep))
        # Pad with -1 so returned tensors have the same shape
        gap = config.DETECTION_MAX_INSTANCES - K.shape(class_keep)[0]
        class_keep = tf.pad(class_keep, [(0, gap)], 
                           mode='CONSTANT', constant_values=-1)
        # Set shape so map_fn() can infer result shape
        class_keep.set_shape([config.DETECTION_MAX_INSTANCES])
        return class_keep

    # Map over class IDs
    nms_keep = K.map_fn(nms_keep_map, unique_pre_nms_class_ids, dtype='int64')
    # Merge results into one list and remove padding
    nms_keep = K.reshape(nms_keep, [-1])
    nms_keep = K.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
    # Compute intersection between keep and nms_keep
    keep = tf.sets.set_intersection(K.expand_dims(keep, 0),
                                    K.expand_dims(nms_keep, 0))
    keep = K.to_dense(keep)[0]
    # Keep top detections
    roi_count = config.DETECTION_MAX_INSTANCES
    class_scores_keep = K.gather(class_scores, keep)
    num_keep = K.minimum(K.shape(class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    keep = K.gather(keep, top_ids)

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    detections = K.concatenate([
        K.gather(refined_rois, keep),
        tf.to_float(K.gather(class_ids, keep))[..., tf.newaxis],
        tf.gather(class_scores, keep)[..., tf.newaxis]], axis=1)
    
    # Pad with zeros if detections < DETECTION_MAX_INSTANCES
    gap = config.DETECTION_MAX_INSTANCES - K.shape(detections)[0]
    detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
    return detections

class DetectionLayer(KE.Layer):
    '''
    Takes classified proposal boxes and their bounding box deltas
    and returns the final detection boxes

    Returns
        [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] with
        normalized coords
    '''

    def __init__(self, config=None, **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        rois = inputs[0]
        frcnn_class = inputs[1]
        frcnn_bbox = inputs[2]
        image_meta = inputs[3]

        # Get windows of images in normalized coordinates. Windows are the
        # area in the image that excludes the padding.
        m = parse_image_meta_graph(image_meta)
        image_shape = m['image_shape'][0]
        window = norm_boxes_graph(m['window'], image_shape[:2])

        # Run detection refinement on each item in the batch
        detections_batch = utils.batch_slice(
            [rois, frcnn_class, frcnn_bbox, window],
            lambda w, x, y, z: refine_detections_graph(w, x, y, z, self.config),
            self.config.IMAGES_PER_GPU)

        # Reshape output
        return K.reshape(detections_batch,
            [self.config.BATCH_SIZE, self.config.DETECTION_MAX_INSTANCES, 6])

    def compute_output_shape(self, input_shape):
        return (None, self.config.DETECTION_MAX_INSTANCES, 6)

#################################
# Region Proposal Network (RPN) #
#################################

def rpn_graph(feature_map, anchors_per_location, anchor_stride):
    '''
    Builds the computation graph of the Region Proposal Network
    Arguments
        feature_map: [batch, height, width, depth] backbone featurs
        anchors_per_location: number of anchors per pixel in the feature map
        anchor_stride: Controls the density of the anchors, usually 1 (anchors
                       for each pixel in the map), or 2 (for every other pixel)
    Returns
        rpn_logits: [batch, H, W, 2] Anchor classifier logits
        rpn_probs: [batch, H, W, 2] Anchor classifier probabilities
        rpn_bbox: [batch, H, W, (dy, dx, log(dh), log(dw))] Deltas to be applied
                  to anchors
    '''

    # Shared convolutional base of the RPN
    shared = Conv2D(512, (3, 3), padding='same', activation='relu', 
        strides=anchor_stride, name='rpn_conv_shared')(feature_map)

    # Anchor score [batch, height, width, anchors_per_location * 2]
    x = Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
        activation='linear', name='rpn_class_raw')(shared)

    # Reshape to [batch, anchors, 2]
    rpn_class_logits = Lambda(lambda t: K.reshape(t, [K.shape(t)[0], -1, 2]))(x)

    # Softmax on last dimension of BG/FG
    rpn_probs = Activation("softmax", name="rpn_class_xxx")(rpn_class_logits)

    # Bounding box refinement
    x = Conv2D(anchors_per_location * 4, (1, 1), padding='valid',
        activation='linear', name='rpn_bbox_pred')(shared)

    # Reshape to [batch, anchors, 4]
    rpn_bbox = Lambda(lambda t: K.reshape(t, [K.shape(t)[0], -1, 4]))(x)

    return [rpn_class_logits, rpn_probs, rpn_bbox]

def build_rpn_model(anchor_stride, anchors_per_location, depth):
    '''
    Builds a Keras model of the Region Proosal Network. The RPN graph is
    wrapped so it can be used multiple times with shared weights.
    Arguments
        anchors_per_location: Number of anchors per pixel in the feature map
        anchor_stride: Controls the density of anchors, typically 1 or 2
        depth: Depth of the backbone feature map
    Returns
        Keras model object, whose outputs are:
            rpn_logits: [batch, H, W, 2] Anchor classifier logits
            rpn_probs: [batch, H, W, 2] Anchor classifier probabilities
            rpn_bbox: [batch, H, W, (dy, dx, log(dh), log(dw))] Deltas to be 
                      applied to anchors
    '''
    input_feature_map = Input(shape=[None, None, depth], name="input_rpn_feature_map")
    outputs = rpn_graph(input_feature_map, anchors_per_location, anchor_stride)
    return Model([input_feature_map], outputs, name="rpn_model")

#################################
# Feature Pyramid Network Heads #
#################################

def fpn_classifier_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, train_bn=True):
    '''
    Builds the computation graph of the feature pyramid network classifier and
    regressor heads
    Arguments
        rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
              coords
        feature_maps: [P2, P3, P4, P5] List of feature maps from different layers
                      of the pyramid, each with a different resolution
        image_meta: [batch, (meta_data)] Image metadata, see compose_image_meta()
        pool_size: The width of the square feature map generated from ROI pooling
        num_classes: Number of classes, determines the depth of the results
        train_bn: Boolean, train or freeze batch norm layers
    Returns
        logits: [N, NUM_CLASSES] classifier logits
        probs: [N, NUM_CLASSES] classifier probabilities
        bbox_deltas: [N, (dy, dx, log(dh), log(dw))] Deltas to apply to proposals
    '''

    # ROI Pooling
    # Shape: [batch, num_boxes, pool_height, pool_width, channels]
    x = PyramidROIAlign([pool_size, pool_size],
        name="roi_align_classifier")([rois, image_meta] + feature_maps)
    
    # Two 1024 FC layers
    x = TimeDistributed(Conv2D(1024, (pool_size, pool_size)), 
        name="frcnn_class_conv1")(x)
    x = TimeDistributed(BatchNorm(), name="frcnn_class_bn1")(x, training=train_bn)
    x = Activation("relu")(x)

    x = TimeDistributed(Conv2D(1024, (1, 1)), name="frcnn_class_conv2")(x)
    x = TimeDistributed(BatchNorm(), name="frcnn_class_bn2")(x, training=train_bn)
    x = Activation("relu")(x)

    shared = Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2), name="pool_squeeze")(x)

    # Classifier head
    frcnn_class_logits = TimeDistributed(Dense(num_classes),
        name="frcnn_class_logits")(shared)
    frcnn_probs = TimeDistributed(Activation("softmax"),
        name="frcnn_class")(frcnn_class_logits)

    # Bounding Box head
    x = TimeDistributed(Dense(num_classes * 4, activation='linear'),
        name='frcnn_bbox_fc')(shared)
    # Reshape
    s = K.int_shape(x)
    frcnn_bbox = Reshape((s[1], num_classes, 4), name="frcnn_bbox")(x)

    return frcnn_class_logits, frcnn_probs, frcnn_bbox

def build_fpn_mask_graph(rois, feature_maps, image_meta, 
                         pool_size, num_classes, train_bn=True):
    '''
    Builds the computation graph of the mask head of the Feature Pyramid Network
    Arguments
        rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized coords
        feature_maps: [P2, P3, P4, P5] List of feature maps from different layers
                      of the pyramid.
        image_meta: [batch, (meta_data)] Image details, see compose_image_meta()
        pool_size: The wdith of the square feature map generated from ROI pooling
        num_classes: The number of classes, determines depth of results
        train_bn: Boolean, train or freeze batch norm layers
    Returns
        masks: [batch, roi_count, height, width, num_classes]
    ''' 
    # Not implemented, method used for Mask RCNN


##################
# Loss Functions #
##################

def smooth_l1_loss(y_true, y_pred):
    '''
    Implements smooth-l1 loss.
    y_true and y_pred are typically: [N, 4] but could be any shpae 
    '''
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return loss

def rpn_class_loss_graph(rpn_match, rpn_class_logits):
    '''
    RPN anchor classifier loss
    Arguments
        rpn_match: [batch, anchors, 1] Anchor match type,
                   1=positive, -1=negative, 0=neutral
        rpn_class_logits: [batch, anchors, 2] RPN classifier logits for FG/BG
    '''
    # Squeeze last dim
    rpn_match = K.squeeze(rpn_match, -1)
    # Get anchor classes, convert the +/- 1 match to 0/1 values
    anchor_class = K.cast(K.equal(rpn_match, 1), 'int32')
    # Positive and Negative anchors contribute to the loss, but neutral
    # anchors do not
    ix = tf.where(K.not_equal(rpn_match, 0))
    # Filter out rows that do not contribute to the loss
    rpn_class_logits = tf.gather_nd(rpn_class_logits, ix)
    anchor_class = tf.gather_nd(anchor_class, ix)
    # Crossentropy loss
    loss = K.sparse_categorical_crossentropy(target=anchor_class,
                                             output=rpn_class_logits,
                                             from_logits=True)
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), K.constant(0.))
    return loss

def rpn_bbox_loss_graph(config, target_bbox, rpn_match, rpn_bbox):
    '''
    Return the RPN bounding box loss graph\
    Arguments
        config: The model config object
        target_bbox: [batch, max_positive_anchors, (dy, dx, log(dh), log(dw))]
                     Uses zero padding to fill in unsed box deltas
        rpn_match: [batch, anchors, 1] Anchor match type,
                 1=positive, -1=negative, 0=neutral
        rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    '''
    # Filter out netural anchors
    rpn_match = K.squeeze(rpn_match, -1)
    ix = tf.where(K.equal(rpn_match, 1))
    rpn_bbox = tf.gather_nd(rpn_bbox, ix)

    # Trim target bounding box deltas to the same length as rpn_bbox
    batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), 'int32'), axis=1)
    target_bbox = batch_pack_graph(target_bbox, batch_counts, config.IMAGES_PER_GPU)

    loss = smooth_l1_loss(target_bbox, rpn_bbox)
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), K.constant(0.))
    return loss

def frcnn_class_loss_graph(target_class_ids, pred_class_logits, active_class_ids):
    '''
    Loss for the classifier head of Faster RCNN
    Arguments
        target_class_ids: [batch, num_rois] Integer class IDs with zero padding
        pred_class_logits: [batch, num_rois, num_classes]
        active_class_ids: [batch, num_classes] Has a value of 1 for IDs in the dataset
                          and 0 otherwise
    '''
    target_class_ids = K.cast(target_class_ids, 'int64')

    # Find predctions of classes that are not in the dataset
    pred_class_ids = K.argmax(pred_class_logits, axis=2)
    pred_active = K.gather(active_class_ids[0], pred_class_ids)

    # Loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_class_ids, logits=pred_class_logits)
    loss = loss * pred_active

    loss = K.sum(loss) / K.sum(pred_active)
    return loss

def frcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
    '''
    Loss for Faster RCNN bounding box refinement
    Arguments
        target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
        target_class_ids: [batch_num_rois] Integer class ids
        pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    '''

    # Reshape to merge batch and ROI dimensions
    target_class_ids = K.reshape(target_class_ids, (-1,))
    target_bbox = K.reshape(target_bbox, (-1, 4))
    pred_bbox = K.reshape(pred_bbox, (-1, K.int_shape(pred_bbox)[2], 4))

    # Filter out non-positive ROIs and incorrect ROI class IDs
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = K.cast(
        K.gather(target_class_ids, positive_roi_ix), 'int64')
    ix = K.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    # Gather the deltas that contribute to loss
    target_bbox = K.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather_nd(pred_bbox, ix)

    # Smooth L1 loss
    loss = K.switch(tf.size(target_bbox) > 0, 
        smooth_l1_loss(target_bbox, pred_bbox), K.constant(0.))
    loss = K.mean(loss)
    
    return loss 

###################
# Data generators #
###################

def load_image_gt(dataset, config, image_id, augmentation=None):
    # TODO: This method loads in bounding boxes from masks, rewrite so that
    # no code depends on masks
    '''
    Load and return ground truth data for an image (image, bounding boxes)
    Arguments
        dataset: The dataset for the image
        config: Configuration defined in utils/config.py
        image_id: Integer ID for the image
        augmentation: Optional, specify an imgaug object to transform images
                      see https://github.com/aleju/imgaug for more info
    Returns
        image: [H, W, 3]
        shape: The original shape of the image before resizing/cropping
        class_ids: [instance_count] Integer class IDs
        bbox: [instance_count, (y1, x1, y2, x2)]        
    '''

    # Load image and mask
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    original_shape = image.shape
    image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)
    
    # Augmentation (Requires the imgaug library at https://github.com/aleju/imgaug)
    if augmentation:
        import imgaug

        # Augmentors that are safe to apply to masks
        # Note: Affine and PiecewiseAffine need more testing
        MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                           "Fliplr", "Flipud", "CropAndPad",
                           "Affine", "PiecewiseAffine"]

        def hook(images, augmenter, parents, default):
            # Determines which augmenters to apply to masks
            return (augmenter.__class__.__name__ in MASK_AUGMENTERS)

        # Store shapes before augmentation to compare
        image_shape = image.shape
        mask_shape = mask.shape

        # Make augmenters deterministic
        det = augmentation.to_deterministic()
        image = det.augment_image(image)
        # Change mask to np.uint8 because imgaug doesn't support np.bool
        mask = det.augment_image(mask.astype(np.uint8), 
                                 hooks=imgaug.HooksImages(activator=hook))
        
        # Verify that shapes didn't change
        assert image.shape == image_shape, "Augmentation shouldn't change image size"
        assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
        # Change mask back to bool
        mask = mask.astype(np.bool)
    
    # Some boxes might be all zeros if the corresponding mask was cropped out,
    # so filter them out
    _idx = np.sum(mask, axis=(0, 1)) > 0
    mask = mask[:, :, _idx]
    class_ids = class_ids[_idx]
    # Bounding boxes
    bbox = utils.extract_bboxes(mask)

    # Active classes
    # Different datasets have different classes, so track the 
    # classes supported in the dataset of this image
    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    active_class_ids[source_class_ids] = 1

    # Image meta data
    image_meta = compose_image_meta(image_id, original_shape, image.shape,
                                    window, scale, active_class_ids)

    return image, image_meta, class_ids, bbox

def build_detection_targets(rpn_rois, gt_class_ids, gt_boxes, config):
    '''
    Generate targets for training Stage 2 classifier and frcnn heads.
    This is not used in normal training. It's useful for debugging or to train
    the Faster RCNN heads without using the RPN head.
    Inputs:
        rpn_rois: [N, (y1, x1, y2, x2)] proposal boxes.
        gt_class_ids: [instance count] Integer class IDs
        gt_boxes: [instance count, (y1, x1, y2, x2)]
    Returns:
        rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)]
        class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
        bboxes: [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (y, x, log(h), log(w))]. Class-specific
                bbox refinements.
    '''
    assert rpn_rois.shape[0] > 0
    assert gt_class_ids.dtype == np.int32, "Expected int but got {}".format(
        gt_class_ids.dtype)
    assert gt_boxes.dtype == np.int32, "Expected int but got {}".format(
        gt_boxes.dtype)

    # Trim empty padding in gt_boxes and gt_masks parts
    instance_ids = np.where(gt_class_ids > 0)[0]
    assert instance_ids.shape[0] > 0, "Image must contain instances."
    gt_class_ids = gt_class_ids[instance_ids]
    gt_boxes = gt_boxes[instance_ids]

    # Compute areas of ROIs and ground truth boxes.
    rpn_roi_area = (rpn_rois[:, 2] - rpn_rois[:, 0]) * \
        (rpn_rois[:, 3] - rpn_rois[:, 1])
    gt_box_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * \
        (gt_boxes[:, 3] - gt_boxes[:, 1])

    # Compute overlaps [rpn_rois, gt_boxes]
    overlaps = np.zeros((rpn_rois.shape[0], gt_boxes.shape[0]))
    for i in range(overlaps.shape[1]):
        gt = gt_boxes[i]
        overlaps[:, i] = utils.compute_iou(
            gt, rpn_rois, gt_box_area[i], rpn_roi_area)

    # Assign ROIs to GT boxes
    rpn_roi_iou_argmax = np.argmax(overlaps, axis=1)
    rpn_roi_iou_max = overlaps[np.arange(
        overlaps.shape[0]), rpn_roi_iou_argmax]
    # GT box assigned to each ROI
    rpn_roi_gt_boxes = gt_boxes[rpn_roi_iou_argmax]
    rpn_roi_gt_class_ids = gt_class_ids[rpn_roi_iou_argmax]

    # Positive ROIs are those with >= 0.5 IoU with a GT box.
    fg_ids = np.where(rpn_roi_iou_max > 0.5)[0]

    # Negative ROIs are those with max IoU 0.1-0.5 (hard example mining)
    # TODO: To hard example mine or not to hard example mine, that's the question
    # bg_ids = np.where((rpn_roi_iou_max >= 0.1) & (rpn_roi_iou_max < 0.5))[0]
    bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]

    # Subsample ROIs. Aim for 33% foreground.
    # FG
    fg_roi_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
    if fg_ids.shape[0] > fg_roi_count:
        keep_fg_ids = np.random.choice(fg_ids, fg_roi_count, replace=False)
    else:
        keep_fg_ids = fg_ids
    # BG
    remaining = config.TRAIN_ROIS_PER_IMAGE - keep_fg_ids.shape[0]
    if bg_ids.shape[0] > remaining:
        keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
    else:
        keep_bg_ids = bg_ids
    # Combine indicies of ROIs to keep
    keep = np.concatenate([keep_fg_ids, keep_bg_ids])
    # Need more?
    remaining = config.TRAIN_ROIS_PER_IMAGE - keep.shape[0]
    if remaining > 0:
        # Looks like we don't have enough samples to maintain the desired
        # balance. Reduce requirements and fill in the rest. This is
        # likely different from the Mask RCNN paper.

        # There is a small chance we have neither fg nor bg samples.
        if keep.shape[0] == 0:
            # Pick bg regions with easier IoU threshold
            bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]
            assert bg_ids.shape[0] >= remaining
            keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
            assert keep_bg_ids.shape[0] == remaining
            keep = np.concatenate([keep, keep_bg_ids])
        else:
            # Fill the rest with repeated bg rois.
            keep_extra_ids = np.random.choice(
                keep_bg_ids, remaining, replace=True)
            keep = np.concatenate([keep, keep_extra_ids])
    assert keep.shape[0] == config.TRAIN_ROIS_PER_IMAGE, \
        "keep doesn't match ROI batch size {}, {}".format(
            keep.shape[0], config.TRAIN_ROIS_PER_IMAGE)

    # Reset the gt boxes assigned to BG ROIs.
    rpn_roi_gt_boxes[keep_bg_ids, :] = 0
    rpn_roi_gt_class_ids[keep_bg_ids] = 0

    # For each kept ROI, assign a class_id, and for FG ROIs also add bbox refinement.
    rois = rpn_rois[keep]
    roi_gt_boxes = rpn_roi_gt_boxes[keep]
    roi_gt_class_ids = rpn_roi_gt_class_ids[keep]

    # Class-aware bbox deltas. [y, x, log(h), log(w)]
    bboxes = np.zeros((config.TRAIN_ROIS_PER_IMAGE,
                       config.NUM_CLASSES, 4), dtype=np.float32)
    pos_ids = np.where(roi_gt_class_ids > 0)[0]
    bboxes[pos_ids, roi_gt_class_ids[pos_ids]] = utils.box_refinement(
        rois[pos_ids], roi_gt_boxes[pos_ids, :4])
    # Normalize bbox refinements
    bboxes /= config.BBOX_STD_DEV

    return rois, roi_gt_class_ids, bboxes

def build_rpn_targets(image_shape, anchors, gt_class_ids, gt_boxes, config):
    '''
    Given the anchors and GT boxes, compute overlaps and identify
    positive anchors and deltas to refine them to match their GT boxes
    Arguments
        anchors: [num_anchors, (y1, x1, y2, x2)]
        gt_class_ids: [num_gt_boxes] Integer class IDs
        gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]
    Returns
        rpn_match: [N] matches between anchors and GT boxes,
                   1 = positive anchor, -1 = negative anchor, 0  neutral
        rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas
    '''

    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

    # Handle COCO crowds
    # A crowd box in COCO is a bbox around several instances, and are
    # denoted with negative class IDs. Exclude them from training
    crowd_ix = np.where(gt_class_ids < 0)[0]
    if crowd_ix.shape[0] > 0:
        # Filter corwds from GT class IDs and bboxes
        non_crowd_ix = np.where(gt_class_ids > 0)[0]
        crowd_boxes = gt_boxes[crowd_ix]
        gt_class_ids = gt_class_ids[non_crowd_ix]
        gt_boxes = gt_boxes[non_crowd_ix]
        # Compute overlaps with crowd boxes
        crowd_overlaps = utils.compute_overlaps(anchors, crowd_boxes)
        crowd_iou_max = np.amax(crowd_overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < 0.001)
    else:
        no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

    # Compute overlaps 
    overlaps = utils.compute_overlaps(anchors, gt_boxes)

    # Match anchors to GT boxes
    # If an anchor overlaps a GT box with IoU >= 0.7 it's positive
    # If an anchor overlaps a GT box with IoU <= 0.3 it's negative
    # Neutral anchors are those that don't match the above conditions,
    # and they have no effect on the loss function
    # No GT box should be unmatched, instead match them to the closest anchor

    # Set negative anchors first
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1
    # Set an anchor for each GT box (regardless of IoU value)
    gt_iou_argmax = np.argmax(overlaps, axis=0)
    rpn_match[gt_iou_argmax] = 1
    # Set anchors with high overlap as positive
    rpn_match[anchor_iou_max >= 0.7] = 1

    # Subsample to balance positive and negative anchors
    # Don't let positives be more tan half the anchors
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE - np.sum(rpn_match == 1))

    if extra > 0:
        # Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    
    # For positive anchors, compute shift and scale needed to transform them
    # to match the correspodning GT boxes
    ids = np.where(rpn_match == 1)[0]
    ix = 0
    # TODO: The following is duplicated frfom box_refinement()
    for i, a in zip(ids, anchors[ids]):
        # Closest GT box
        gt = gt_boxes[anchor_iou_argmax[i]]

        # Convert coordinates to center + W/H
        # GT box
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        # Anchor
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        # Compute the BBox refinement that the RPN should predict
        rpn_bbox[ix] = [
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w,
            np.log(gt_h / a_h),
            np.log(gt_w / a_w)
        ]
        # Normalize
        rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1
    
    return rpn_match, rpn_bbox

def generate_random_rois(image_shape, count, gt_class_ids, gt_boxes):
    '''
    Generates ROI proposals similar to what a region proposal network would generate
    Arguments
        image_shape: [H, W, depth]
        count: Number of ROIs to generate
        gt_class_ids: [N] Integer GT class IDs
        gt_boxes: [N, (y1, x1, y2, x2)] Ground truth boxes in pixel coords
    Returns
        rois: [count, (y1, x1, y2, x2)] ROI boxes in pixels
    '''
    rois = np.zeros((count, 4), dtype=np.int32)
    #y1y2 = np.zeros((1, 2), dtype=np.int32)
    #x1x2 = np.zeros((1, 2), dtype=np.int32)

    # Generate random ROIs around GT boxes (90% of count)
    rois_per_box = int(0.9 * count / gt_boxes.shape[0])
    for i in range(gt_boxes.shape[0]):
        gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[i]
        h = gt_y2 - gt_y1
        w = gt_x2 - gt_x1
        # Random boundaries
        r_y1 = max(gt_y1 - h, 0)
        r_y2 = min(gt_y2 + h, image_shape[0])
        r_x1 = max(gt_x1 - w, 0)
        r_x2 = min(gt_x2 + w, image_shape[1])

        # To avoid generating boxes with zero area, generate double
        # what is needed and filter out the extra. If we get fewer
        # valid boxes than necessary try again
        # Note: Optimizing this block is unlikely to lead to major performance
        # increases as the odds of generating over 50% zero area boxes is low
        while True:
            y1y2 = np.random.randint(r_y1, r_y2, (rois_per_box * 2, 2))
            x1x2 = np.random.randint(r_x1, r_x2, (rois_per_box * 2, 2))
            thresh = 1
            y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >=
                thresh][:rois_per_box]
            x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >=
                thresh][:rois_per_box]
            if y1y2.shape[0] == rois_per_box and x1x2.shape[0] == rois_per_box:
                break

        # Sort on axis 1 to ensure x1 <= x2 and y1 <= y2, then reshape
        y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
        x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
        box_rois = np.hstack([y1, x1, y2, x2])
        rois[(rois_per_box * i):(rois_per_box * (i + 1))] = box_rois

    # Generate random ROIs anywhere in the image (10% of count)
    remaining_count = count - (rois_per_box * gt_boxes.shape[0])
    # As above, generate random rois and filter out empty rois until enough
    # are generated
    while True:
        y1y2 = np.random.randint(r_y1, r_y2, (remaining_count * 2, 2))
        x1x2 = np.random.randint(r_x1, r_x2, (remaining_count * 2, 2))
        thresh = 1
        y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >=
            thresh][:remaining_count]
        x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >=
            thresh][:remaining_count]
        if y1y2.shape[0] == remaining_count and x1x2.shape[0] == remaining_count:
            break

    # Sort on axis 1 to ensure x1 <= x2 and y1 <= y2, reshape
    y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
    x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
    global_rois = np.hstack([y1, x1, y2, x2])
    rois[-remaining_count] = global_rois
    return rois

def data_generator(dataset, config, shuffle=True, augmentation=None, random_rois=0,
                   batch_size=1, detection_targets=False):
    '''
    A generator that returns images, corresponding class ids, and bounding box deltas
    Arguments
        dataset: The Dataset object to pick data from
        config: The model config object
        shuffle: If true, shuffles the samples before every epoch
        augmentation: Optional imgaug object to transform images.
                      See (https://github.com/aleju/imgaug) for more
        random_rois: If > 0 then generate proposals to be used to train the
                     network classifier.
        batch_size: How many images to return each call
        detection_targets: If true, generate detection targets (class IDs, and bbox deltas)
    Returns
        A Python generator. Each time next() is called, the generator returns two lists,
        inputs and outputs. The contents of the lists differ based on the received args:
        inputs list:
            images: [batch, H, W, C]
            image_meta: [batch, (meta_data)] Image details, see compose_image_meta()
            rpn_match: [batch, N] integer (1=positive anchor, -1=negative, 0=neutral)
            rpn_bbopx: [batch, N, (dy, dx, log(dh), log(dw))] Anchor box deltas
            gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
            gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
        outputs list:
            Usually empty, but if detection_targets is True then the outputs list contains
            target class_ids, and bbox deltas  
    '''
    # Initialize generator
    batch_ix = 0
    image_ix = -1
    image_ids = np.copy(dataset.image_ids)
    error_count = 0

    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    backbone_shapes = compute_backbone_shapes(config, config.IMAGE_SHAPE)
    anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                             config.RPN_ANCHOR_RATIOS,
                                             backbone_shapes,
                                             config.BACKBONE_STRIDES,
                                             config.RPN_ANCHOR_STRIDE)

    # For keras to run indefinitely it needs a data generator
    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch
            image_ix = (image_ix + 1) % len(image_ids)
            if shuffle and image_ix == 0:
                np.random.shuffle(image_ids)

            # Get GT bounding boxes for image
            image_id = image_ids[image_ix]
            image, image_meta, gt_class_ids, gt_boxes = load_image_gt(dataset, 
                config, image_id, augmentation=augmentation)

            # Skip images that have no instances, happens if we train on a
            # subset of classes and the image doesn't have any of those classes
            if not np.any(gt_class_ids > 0):
                continue

            # RPN targets
            rpn_match, rpn_bbox = build_rpn_targets(image.shape, anchors, 
                                                    gt_class_ids, gt_boxes, config)

            # Faster R-CNN Targets
            if random_rois:
                rpn_rois = generate_random_rois(
                    image.shape, random_rois, gt_class_ids, gt_boxes)
                if detection_targets:
                    rois, frcnn_class_ids, frcnn_bbox = build_detection_targets(
                        rpn_rois, gt_class_ids, gt_boxes, config)

            # Init batch arrays
            if batch_ix == 0:
                batch_image_meta = np.zeros(
                    (batch_size,) + image_meta.shape, dtype=image_meta.dtype)
                batch_rpn_match = np.zeros(
                    [batch_size, anchors.shape[0], 1], dtype=rpn_match.dtype)
                batch_rpn_bbox = np.zeros(
                    [batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=rpn_bbox.dtype)
                batch_images = np.zeros(
                    (batch_size,) + image.shape, dtype=np.float32)
                batch_gt_class_ids = np.zeros(
                    (batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)
                batch_gt_boxes = np.zeros(
                    (batch_size, config.MAX_GT_INSTANCES, 4), dtype=np.int32)

                if random_rois:
                    batch_rpn_rois = np.zeros(
                        (batch_size, rpn_rois.shape[0], 4), dtype=rpn_rois.dtype)
                    if detection_targets:
                        batch_rois = np.zeros(
                            (batch_size,) + rois.shape, dtype=rois.dtype)
                        batch_frcnn_class_ids = np.zeros(
                            (batch_size,) + frcnn_class_ids.shape, dtype=frcnn_class_ids.dtype)
                        batch_frcnn_bbox = np.zeros(
                            (batch_size,) + frcnn_bbox.shape, dtype=frcnn_bbox.dtype)
            
            # If more instances than fits in array, sub-sample from them
            if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
                ids = np.random.choice(
                    np.arange(gt_boxes.shape[0]), config.MAX_GT_INSTANCES, replace=False)
                gt_class_ids = gt_class_ids[ids]
                gt_boxes = gt_boxes[ids]

            # Add to batch
            batch_image_meta[batch_ix] = image_meta
            batch_rpn_match[batch_ix] = rpn_match[:, np.newaxis]
            batch_rpn_bbox[batch_ix] = rpn_bbox
            batch_images[batch_ix] = mold_image(image.astype(np.float32), config)
            batch_gt_class_ids[batch_ix, :gt_class_ids.shape[0]] = gt_class_ids
            batch_gt_boxes[batch_ix, :gt_boxes.shape[0]] = gt_boxes
            if random_rois:
                batch_rpn_rois[batch_ix] = rpn_rois
                if detection_targets:
                    batch_rois[batch_ix] = rois
                    batch_frcnn_class_ids[batch_ix] = frcnn_class_ids
                    batch_frcnn_bbox[batch_ix] = frcnn_bbox
            batch_ix += 1

            # Check if batch is full
            if batch_ix >= batch_size:
                inputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,
                          batch_gt_class_ids, batch_gt_boxes]
                outputs = []

                if random_rois:
                    inputs.extend([batch_rpn_rois])
                    if detection_targets:
                        inputs.extend([batch_rois])
                        # Keras requires input and output to have the same number of dims
                        batch_frcnn_class_ids = np.expand_dims(batch_frcnn_class_ids, -1)
                        outputs.extend([batch_frcnn_class_ids, batch_frcnn_bbox])
                
                yield inputs, outputs
                
                # Start a new batch
                batch_ix = 0

        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log the exception and skip the image
            logging.exception("Error processing image {}".format(
                dataset.image_info[image_id]))
            error_count += 1
            if error_count > 5:
                raise


#####################
# Faster RCNN Model #
#####################

class FasterRCNN():

    def __init__(self, mode, config, model_dir):
        '''
        Arguments
            mode: Either 'training' or 'inference'
            config: Configuration defined in utils/configs/FasterRCNNConfig.py
            model_dir: Directory to save logs and weights
        '''
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(mode=mode, config=config)

    def build(self, mode, config):
         # Defining input tensors
        input_image = Input(shape=[None, None, 3], name="input_image")
        input_image_meta = Input(shape=[config.IMAGE_META_SIZE], name="input_image_meta")
        
        if mode == 'training':
            # RPN ground truth inputs
            input_rpn_match = Input(shape=[None, 1], name="input_rpn_match")
            input_rpn_bbox = Input(shape=[None, 4], name="input_rpn_bbox")

            # Detection ground truth
            input_gt_class_ids = Input(shape=[None], name="input_gt_class_ids")
            input_gt_boxes = Input(shape=[None, 4], name="input_gt_boxes")
            # Normalize coords
            gt_boxes = Lambda(lambda x: norm_boxes_graph(x, K.shape(input_image)[1:3]))(input_gt_boxes)
        else:
            # Anchor inputs
            input_anchors = Input(shape=[None, 4], name="input_anchors")

        # Building Feature Pyramid Network
        # Bottom up layers
        _, C2, C3, C4, C5 = resnet_graph(input_image, stage5=True, train_bn=config.TRAIN_BN)

        # Top down layers
        P5 = Conv2D(256, (1, 1), name="fpn_c5p5")(C5)
        P4 = layers.Add(name="fpn_p4add")([
            UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
            Conv2D(256, (1, 1), name="fpn_c4p4")(C4)])
        P3 = layers.Add(name="fpn_p3add")([
            UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
            Conv2D(256, (1, 1), name="fpn_c3p3")(C3)])
        # NOTE: P2's UpSampling2D layer size was temporarily changed from (2, 2) -> (1, 1)
        P2 = layers.Add(name="fpn_p2add")([
            UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
            Conv2D(256, (1, 1), name="fpn_c2p2")(C2)])

        P2 = Conv2D(256, (3, 3), padding="same", name="fpn_p2")(P2)
        P3 = Conv2D(256, (3, 3), padding="same", name="fpn_p3")(P3)
        P4 = Conv2D(256, (3, 3), padding="same", name="fpn_p4")(P4)
        P5 = Conv2D(256, (3, 3), padding="same", name="fpn_p5")(P5)
        P6 = MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)

        rpn_feature_maps = [P2, P3, P4, P5, P6]
        frcnn_feature_maps = [P2, P3, P4, P5]

        # Anchors
        if mode == "training":
            anchors = self.get_anchors(config.IMAGE_SHAPE)
            anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
            anchors = Lambda(lambda x: K.variable(anchors), name="anchors")(input_image)
        else:
            anchors = input_anchors

        # RPN Model
        rpn = build_rpn_model(config.RPN_ANCHOR_STRIDE, 
                len(config.RPN_ANCHOR_RATIOS), 256)

        # Loop through pyramid layers
        layer_outputs = []
        for p in rpn_feature_maps:
            layer_outputs.append(rpn([p]))        
        # Convert from list of lists of level outputs to
        # list of lists of outputs across levels
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [layers.Concatenate(axis=1, name=n)(list(o))
                    for o, n in zip(outputs, output_names)]

        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coords
        if mode == "training":
            proposal_count = config.POST_NMS_ROIS_TRAINING
        else:
            proposal_count = config.POST_NMS_ROIS_INFERENCE
        rpn_rois = ProposalLayer(
            proposal_count=proposal_count,
            nms_threshold=config.RPN_NMS_THRESHOLD,
            name="ROI",
            config=config)([rpn_class, rpn_bbox, anchors])
        
        if mode == "training":
            # Class ID mask to mark class IDs supported by the
            # dataset the image came from
            active_class_ids = Lambda(
                lambda x: parse_image_meta_graph(x)["active_class_ids"]
            )(input_image_meta)

            if not config.USE_RPN_ROIS:
                # Ignore predicted ROIs and use ROIs provided as input
                input_rois = Input(shape=[config.POST_NMS_ROIS_TRAINING, 4],
                    name="input_roi")
                target_rois = Lambda(lambda x: norm_boxes_graph(
                    x, K.shape(input_image)[1:3]))(input_rois)
            else:
                target_rois = rpn_rois

            # Generate detection targets
            # Subsamples proposals and generates target outputs for training.
            # Note that proposal class IDs, and gt_boxes and returned ROIs
            # are zero padded
            rois, target_class_ids, target_bbox = DetectionTargetLayer(
                config, name="proposal_targets")([target_rois, 
                input_gt_class_ids, gt_boxes]
            )

            # Network heads
            frcnn_class_logits, frcnn_class, frcnn_bbox = fpn_classifier_graph(
                rois, frcnn_feature_maps, input_image_meta, config.POOL_SIZE,
                config.NUM_CLASSES, train_bn=config.TRAIN_BN
            )

            # Include?
            '''
            mrcnn_mask = build_fpn_mask_graph(rois, frcnn_feature_maps,
                                                input_image_meta,
                                                config.MASK_POOL_SIZE,
                                                config.NUM_CLASSES,
                                                train_bn=config.TRAIN_BN)
            '''
            # /Include?
            
            output_rois = Lambda(lambda x: x * 1, name="output_rois")(rois)

            # Losses
            rpn_class_loss = Lambda(lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss")(
                [input_rpn_match, rpn_class_logits])
            rpn_bbox_loss = Lambda(lambda x: rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")(
                [input_rpn_bbox, input_rpn_match, rpn_bbox])
            class_loss = Lambda(lambda x: frcnn_class_loss_graph(*x), name="frcnn_class_loss")(
                [target_class_ids, frcnn_class_logits, active_class_ids])
            bbox_loss = Lambda(lambda x: frcnn_bbox_loss_graph(*x), name="frcnn_bbox_loss")(
                [target_bbox, target_class_ids, frcnn_bbox])

            # Model
            inputs = [input_image, input_image_meta, input_rpn_match, input_rpn_bbox,
                    input_gt_class_ids, input_gt_boxes]
            if not config.USE_RPN_ROIS:
                inputs.append(input_rois)
            outputs = [rpn_class_logits, rpn_class, rpn_bbox, frcnn_class_logits,
                    frcnn_class, frcnn_bbox, rpn_rois, output_rois, rpn_class_loss,
                    rpn_bbox_loss, class_loss, bbox_loss]
        
        else:
            # Network heads
            # Proposal calssifier and BBox regressor heads
            frcnn_class_logits, frcnn_class, frcnn_bbox = fpn_classifier_graph(
                rpn_rois, frcnn_feature_maps, input_image_meta, config.POOL_SIZE,
                config.NUM_CLASSES, train_bn=config.TRAIN_BN
            )

            # Detections
            # output is [batc, num_detections, (y1, x1, y2, x2, class_id, score)]
            # in normalized coords
            detections = DetectionLayer(config, name="frcnn_detection")(
                [rpn_rois, frcnn_class, frcnn_bbox, input_image_meta]
            )

            inputs = [input_image, input_image_meta, input_anchors]
            outputs = [detections, frcnn_class, frcnn_bbox, rpn_rois, rpn_class, rpn_bbox]
        
        return (Model(inputs, outputs, name="Faster_RCNN"))

    def find_last(self):
        '''
        Finds the last checkpoint file of the last trained model in the model dir
        Returns
            log_dir: The directory where events and weights are saved
            checkpoint_path: The path to the last checkpoint file
        '''
        # Get directory names, each corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            return None, None
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("faster-rcnn"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            return dir_name, None
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return dir_name, checkpoint

    def load_weights(self, filepath, by_name=False, exclude=None):
        '''
        Modified version of the corresponding keras function with the addition of
        multi-GPU support and the ability to exclude some layers from loading.
        Arguments
            exclude: List of layer names to exclude
        '''

        import h5py
        from keras.engine import topology

        if exclude:
            by_name=True

        if h5py is None:
            raise ImportError('load_weights() reguires h5py')

        f = h5py.File(filepath, mode='r')
        if "layer_names" not in f.attrs and "model_weight" in f:
            f = f["model_weights"]
        
        # In multi-GPU training, we wrap the model. Get layers of the inner
        # model because they have the weights
        keras_model = self.keras_model
        if hasattr(keras_model, "inner_model"):
            layers = keras_model.inner_model.layers
        else:
            layers = keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            topology.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            topology.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

        # Update thelog directory
        self.set_log_dir(filepath)

    def get_imagenet_weights(self):
        '''
        Dowloads ImageNet trained weights from Keras.
        Returns
            weights_path: Path to weights file
        '''
        from keras.utils.data_utils import get_file
        TF_WEIGHTS_PATH_NO_TOP = "https://github.com/fchollet/deep-learning-models/"\
                                 "releases/download/v0.2/"\
                                 "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
        weights_path = get_file("resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",
                                TF_WEIGHTS_PATH_NO_TOP,
                                cache_subdir="models",
                                md5_hash="a268eb855778b3df3c7506639542a6af")
        return weights_path
    
    def compile(self, learning_rate, momentum):
        '''
        Prepares the model for training (add losses, metrics, etc.)
        then compiles the model
        '''
        # Optimizer
        optimizer = SGD(lr=learning_rate, momentum=momentum,
                        clipnorm=self.config.GRADIENT_CLIP_NORM)

        # Losses
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        loss_names = ["rpn_class_loss", "rpn_bbox_loss",
                        "frcnn_class_loss", "frcnn_bbox_loss"]
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output not in self.keras_model.losses:
                loss = (
                    K.mean(layer.output, keepdims=True) *
                    self.config.LOSS_WEIGHTS.get(name, 1.))
                self.keras_model.add_loss(loss)

        # L2 Regularization
        reg_losses = [
            l2(self.config.WEIGHT_DECAY)(w) / K.cast(np.prod(K.int_shape(w)), 'float32')
            for w in self.keras_model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name]
        self.keras_model.add_loss(tf.add_n(reg_losses))
        
        # OOM debugging
        run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)

        # Compile
        self.keras_model.compile(
            optimizer=optimizer,
            loss=[None] * len(self.keras_model.outputs), options=run_opts)

        # Add loss metrics
        for name in loss_names:
            if name not in self.keras_model.metrics_names:
                layer = self.keras_model.get_layer(name)
                self.keras_model.metrics_names.append(name)
                loss = (
                    K.mean(layer.output, keepdims=True) *
                    self.config.LOSS_WEIGHTS.get(name, 1.))
                self.keras_model.metrics_tensors.append(loss)

    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        '''
        Sets model layers as trainable if their names match the given regular
        expression
        '''
        # Print message on the first call (not on recursive calls)
        if verbose > 0 and keras_model is None:
            log("Selecting layers to train")

        keras_model = keras_model or self.keras_model

        # In multi GPU training, wrap the model. Get layers of the
        # inner model because they have the weights
        if hasattr(keras_model, "inner_model"):
            layers = keras_model.inner_model.layers
        else:
            layers = keras_model.layers
        
        for layer in layers:
            # Check if the layer a model
            if layer.__class__.__name__ == "Model":
                print("In model: ", layer.name)
                self.set_trainable(layer_regex, keras_model=layer, indent=indent + 4)
                continue
            
            if not layer.weights:
                continue
            
            # Check if layer is traiable
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            if layer.__class__.__name__ == "TimeDistributed":
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable

            # Print trainable layer names
            if trainable and verbose > 0:
                log("{}{:20}\t({})".format(" " * indent, layer.name, layer.__class__.__name__))

    def set_log_dir(self, model_path=None):
        '''
        Sets the model log directory and epoch counter
        Arguments
            model_path: If none, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise extract
            the log directory and the epoch counter form the file name
        '''
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a valid model path use it
        if model_path:
            # Continue from where last left off, get epoch and date from the 
            # file name. A sample model path might look like:
            # /path/to/logs/coco20171029T2315/faster_rcnn_coco_0001.h5
            regex = r".*/[\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/faster\_rcnn\_[\w-]+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                # Epoch number in file is 1-based, and in keras code its 0-based, so
                # adjust for that then increment by one to start from the next epoch
                self.epoch = int(m.group(6)) - 1 + 1
                print("Re starting from epoch %d" % (self.epoch))

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        # Create log_dir if it doesn't already exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Path to save after each epoch. Include placeholders that get filled by keras
        self.checkpoint_path = os.path.join(self.log_dir, "faster_rcnn_{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace("*epoch*", "{epoch:04d}") 

    def train(self, train_dataset, val_dataset, learning_rate, 
              epochs, layers, augmentation=None):
        '''
        Trains the model
        Arguments
            train_dataset, val_dataset: Dataset objects for training and validation
            learning_rate: Learning rate for training
            epochs: Total number training epochs
            layers: Allows selecting which layers to train
                    "heads" - All layers excluding the backbone (ResNet50)
                    "all" - All layers
                    "3+" - Train Resnet stage 3 and up
                    "4+" - Train Resnet stage 4 and up
                    "5+" - Train Resnet stage 5 and up
            Augmentation: Optional, allows for custom image augmentations while
                          training. The following will horizontally flip images
                          50% of the time:
                          augmentation = imgaug.augmenters.Sometimes(0.5, [
                              imgaug.augmenters.Fliplr(1)
                          ])

                          See https://github.com/aleju/imgaug for more information
        '''
        
        # Pre-defined regular expressions
        layer_regex = {
            "heads": r"(frcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(frcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(frcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(frcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "all": ".*"
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        train_generator = data_generator(train_dataset, self.config, shuffle=True,
                                         augmentation=augmentation,
                                         batch_size=self.config.BATCH_SIZE)
        val_generator = data_generator(val_dataset, self.config, shuffle=True,
                                       batch_size=self.config.BATCH_SIZE)
        
        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=0,
                                        write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path, verbose=0,
                                            save_weights_only=True)
        ]

        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        if os.name is 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=False
        )
        self.epoch = max(self.epoch, epochs)

    def mold_inputs(self, images):
        '''
        Takes a list of images and modifies them to the format expected
        as input to the neural network.
        Arguments
            images: list of image matrices [h, w, c], images can be any size
        Returns
            molded_images: [N, h, w, 3]. Images resized/normalized
            image_metas: [N, meta data length]
            windows: [N, (y1, x1, y2, x2)] The portion of the image with the
                     original image
        '''
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            molded_image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                min_scale=self.config.IMAGE_MIN_SCALE,
                max_dim=self.config.IMAGE_MAX_DIM,
                mode=self.config.IMAGE_RESIZE_MODE)
            molded_image = mold_image(molded_image, self.config)

            # Get image meta
            image_meta = compose_image_meta(
                0, image.shape, molded_image.shape, window, scale,
                np.zeros([self.config.NUM_CLASSES], dtype=np.int32)            )
            molded_images.append(molded_image)
            image_metas.append(image_meta)
            windows.append(window)
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)

        return molded_images, image_metas, windows

    def unmold_detections(self, detections, original_image_shape,
                          image_shape, window):
        '''
        Reformats the detections of one image from output of the neural net
        Arguments
            detections: [N, (y1, x1, y2, x2, class_id, score)] in normalized coords
            original_image_shape: [H, W, C] Original image shape before resizing
            image_shape: [H, W, C] Shape of the image after resizing/padding
            window: [y1, x1, y2, x2] Pixel coords of box containing the image
                    excluding padding
        Returns
            boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
            class_ids: [N] Integer class IDs for each bounding box
            scores: [N] Float probability scores for the class IDs
        '''

        # Get number of detections
        # Note: Detections array is padded with zeros, this finds
        # the index of the first zero entry
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, and scores
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]

        # Translate normalized coords in the resized image to pixel
        # coords in the original image before resizing
        window = utils.norm_boxes(window, image_shape[:2])
        wy1, wx1, wy2, wx2 = window
        shift = np.array([wy1, wx1, wy1, wx1])
        wh = wy2 - wy1
        ww = wx2 - wx1
        scale = np.array([wh, ww, wh, ww])
        # Convert boxes to normalized coords on the window
        boxes = np.divide(boxes - shift, scale)
        # Convert boxes to pixel coords on the original image
        boxes = utils.denorm_boxes(boxes, original_image_shape[:2])

        # Filter out detections with zero area
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            N = class_ids.shape[0]

        return boxes, class_ids, scores

    def detect(self, images, verbose=False):
        '''
        Runs detection on a list of images
        Arguments
            images: List of images of any size
            verbose: Boolean, set to true for additional info logging,
                     default False
        Returns
            results: A list of dicts (1 dict per image) containing:
                rois: [N, (y1, x1, y2, x2)] detection bounding boxes
                class_ids: [N] int class IDs
                scores: [N] float probability scores for the class IDs
        '''
        assert len(images) == self.config.BATCH_SIZE,\
            "len(images) must equal batch size"

        if verbose:
            log("Processing {} images".format(len(images)))
            for image in images:
                log("image", image)

        # Mold inputs to format expected by the network
        molded_images, image_metas, windows = self.mold_inputs(images)

        # Validate image sizes
        # (All images in a batch muyst be of the same size)
        image_shape = molded_images[0].shape
        for img in molded_images[1:]:
            assert img.shape == image_shape,\
                "After resizing, all images must have the same size"
        
        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate anchors across batch dimension
        anchors = np.broadcast_to(
            anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
            log("anchors", anchors)
        
        # Run object detection
        detections, _, _, _, _, _, = self.keras_model.predict(
            [molded_images, image_metas, anchors], verbose=verbose)

        # Process detections
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores = self.unmold_detections(
                detections[i], image.shape, molded_images[i].shape, windows[i])
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores
            })

        return results

    def get_anchors(self, image_shape):
        # Returns anchor pyramid given the image size
        backbone_shapes = compute_backbone_shapes(self.config, image_shape)
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            # Generate anchors
            a = utils.generate_pyramid_anchors(
                self.config.RPN_ANCHOR_SCALES,
                self.config.RPN_ANCHOR_RATIOS,
                backbone_shapes,
                self.config.BACKBONE_STRIDES,
                self.config.RPN_ANCHOR_STRIDE)
            
            self.anchors = a
            self._anchor_cache[tuple(image_shape)] = utils.norm_boxes(a, image_shape[:2])
        
        return self._anchor_cache[tuple(image_shape)]

    def find_trainable_layer(self, layer):
        '''
        If a layer in encapusulated by another layer, this function
        digs through the encapsulation and returns the layer holding
        the weights
        '''
        if layer.__class__.__name__ == "TimeDistributed":
            return self.find_trainable_layer(layer.layer)
        return layer
    
    def get_trainable_layers(self):
        '''
        Returns a list of layers that have weights
        '''
        layers = []
        for l in self.keras_model.layers:
            # If layer is a wrapper, find inner traiable layer
            l = self.find_trainable_layer(l)
            # Include layer if it has weights
            if l.get_weights():
                layers.append(l)
        return layers

###################
# Data Formatting #
###################

def compose_image_meta(image_id, original_image_shape, image_shape,
                       window, scale, active_class_ids):
    '''
    Takes attributes of an image and puts then a 1D array
    Arguments:
        image_id: An int ID of the image (for debugging)
        original_image_shape: [H, W, C] before resizing/padding
        image_shape: [H, W, C] after resizing/padding
        window: (y1, x1, y2, x2) in pixels. The area of the image 
                containing the real image (exluding padding)
        scale: Float, the scaling factor applied to the original image
        active_class_ids: List of class_ids available in the images source
                         dataset, used if training on multiple datasets
    Returns:
        meta: A 1D array containing the images metadata
    '''
    return np.array(
        [image_id] + # meta[0]
        list(original_image_shape) + # meta[1:4]
        list(image_shape) + # meta[4:7]
        list(window) + # meta[7:11]
        [scale] + # meta[11]
        list(active_class_ids) # meta[12:]
    )

def parse_image_meta(meta):
    '''
    Parses an array containing image metadata
    For metadata details see compose_image_data()
    Arguments
        meta: [batch, meta_length] where meta_length depends on NUM_CLASSES
    Returns
        A dict of the parsed values
    '''
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]
    return {
        "image_id": image_id.astype(np.int32),
        "original_image_shape": original_image_shape.astype(np.int32),
        "image_shape": image_shape.astype(np.int32),
        "window": window.astype(np.int32),
        "scale": scale.astype(np.float32),
        "active_class_ids": active_class_ids.astype(np.int32)
    }

def parse_image_meta_graph(meta):
    '''
    Parses a tensor that contains image attributes to its components
    For metadata details see compose_image_meta()
    Arguments
        meta: [batch, meta_length] where meta_length depends on NUM_CLASSES
    Returns
        A dict of the parsed tensors
    '''
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]
    return {
        "image_id": image_id,
        "original_image_shape": original_image_shape,
        "image_shape": image_shape,
        "window": window,
        "scale": scale,
        "active_class_ids": active_class_ids,
    }

def mold_image(images, config):
    '''
    Takes an RGB image (or array of images), subtracts the mean pixel
    and converts to float
    '''
    return images.astype(np.float32) - config.MEAN_PIXEL

def unmold_image(normalized_images, config):
    '''
    Takes an image (or array of images) normalized with mold_images()
    and returns the original
    '''
    return (normalized_images + config.MEAN_PIXEL).astype(np.uint8)

##################
# Misc Functions #
##################

def trim_zeros_graph(boxes, name=None):
    '''
    Function that removes zero padding from bounding box matricies
    Arguments
        boxes: [N, 4] a matrix of boxes
    Returns
        boxes: [N, 4] a matrix of boxes
        non_zeros: [N] a 1D boolean mask identifying the rows to keeo
    '''

    # TODO: boolean_mask does not exist in keras backend, make
    # custom method to remove all zero padding boxes
    non_zeros = tf.cast(K.sum(K.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros

def batch_pack_graph(x, counts, num_rows):
    '''
    Picks different number of values from each row in x
    depending on the value in counts
    '''
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return K.concatenate(outputs, axis=0)

def norm_boxes_graph(boxes, shape):
    '''
    Converts boxes from pixel coordinates to normalized coordinates.
    Arguments
        boxes: [..., (y1, x1, y2, x2)] in pixel coordinates
        shape: [..., (height, width)] in pixels
    Returns:
        [..., (y1, x1, y2, x2)] in normalized coordinates
    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.
    '''
    #shape_fl = K.cast(shape, "float32")
    #h = shape_fl[:, 0]
    #w = shape_fl[:, 1]
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = K.concatenate([h, w, h, w], axis=-1) - K.constant(1.0)
    shift = K.constant([0., 0., 1., 1.])
    return tf.divide(boxes - shift, scale)


def denorm_boxes_graph(boxes, shape):
    '''
    Converts boxes from normalized coordinates to pixel coordinates.
    Arguments
        boxes: [..., (y1, x1, y2, x2)] in normalized coordinates
        shape: [..., (height, width)] in pixels
    Returns:
        [..., (y1, x1, y2, x2)] in pixel coordinates
    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.
    '''
    #shape_fl = K.cast(shape, "float32")
    #h = shape_fl[:, 0]
    #w = shape_fl[:, 1]
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = K.concatenate([h, w, h, w], axis=-1) - K.constant(1.0)
    shift = K.constant([0., 0., 1., 1.])
    return K.cast(K.round(K.dot(boxes, scale) + shift), 'int32')
# Changed file