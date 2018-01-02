"""
implementation of R-FCN Detector
"""
import os
import numpy as np
from PIL import Image
import argparse
import time
import json
from pprint import pprint
import cv2

import os
import sys
os.environ['GLOG_minloglevel'] = '2'
lib_path = './lib'
sys.path.insert(0, lib_path)
print('fast_rcnn lib path add done!')

from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
from vis.visual_kit import combine_detections


CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')


class RFCNDetector(object):

    def __init__(self, proto_file, caffemodel_file, debug=False):
        self.proto_file = proto_file
        self.caffemodel_file = caffemodel_file
        self.debug = debug

        self._init_net()

    def _init_net(self):
        print('proto:', self.proto_file)
        print('caffemodel:', self.caffemodel_file)
        caffe.set_mode_gpu()
        caffe.set_device(0)
        cfg.TEST.HAS_RPN = True
        self.net = caffe.Net(self.proto_file, self.caffemodel_file, caffe.TEST)
        print('caffe net init done!')

    def detect_on_img(self, img_array):
        assert isinstance(img_array, np.ndarray), 'image array must be numpy array'
        print(img_array.shape)
        timer = Timer()
        timer.tic()
        scores, boxes = im_detect(self.net, img_array)
        timer.toc()
        print('Detection took {:.3f}s for '
              '{:d} object proposals').format(timer.total_time, boxes.shape[0])

        # Visualize detections for each class
        CONF_THRESH = 0.8
        NMS_THRESH = 0.3
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1  # because we skipped background
            cls_boxes = boxes[:, 4:8]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            # cls_boxes = dets[:, :-2]
            # print(cls_boxes.shape)
            # cls_scores = dets[:, -1]
            # print(cls_scores.shape)
            # cls_index = np.full(cls_scores.shape, cls_ind)
            # print(cls_index.shape)
            # dets = combine_detections(cls_index, cls_boxes, cls_scores)
            # print(dets.shape)

            # print('detections: ', dets)
            print(dets.shape)
            self.vis_detections(img_array, cls, dets, thresh=CONF_THRESH)
        plt.show()
        # visualize_det_cv2(img_array)

    def detect_on_video(self, video_f, record=False, save_img=True, is_show=True):
        if os.path.exists(video_f):
            cap = cv2.VideoCapture(video_f)
            i = 0
            while cap.isOpened():
                ret, frame = cap.read()
                tic = time.time()
                if ret:
                    i += 1
                    pass

    def detect_on_image_list(self, img_list, is_show=True):
        for img_f in img_list:
            if os.path.exists(img_f):
                img = cv2.imread(img_f)
                self.detect_on_img(img)
            else:
                print('Passing not exist file: ', img_f)

    @staticmethod
    def vis_detections(im, class_name, dets, thresh=0.5):
        """Draw detected bounding boxes."""
        # plt.clf()
        inds = np.where(dets[:, -1] >= thresh)[0]
        if len(inds) == 0:
            return

        im = im[:, :, (2, 1, 0)]
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(im, aspect='equal')
        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]

            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='red', linewidth=3.5)
            )
            ax.text(bbox[0], bbox[1] - 2,
                    '{:s} {:.3f}'.format(class_name, score),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=14, color='white')

        ax.set_title(('{} detections with '
                      'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                      thresh),
                     fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.draw()