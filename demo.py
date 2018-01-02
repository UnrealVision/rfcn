#!/usr/bin/env python

"""
My implementation of demo on R-FCN,
We will test on videos and images.
"""
import os
import argparse

from rfcn_detector import RFCNDetector


def parse_args():
    arg_parser = argparse.ArgumentParser('demo for object detection API.')
    arg_parser.add_argument('--proto_f', default='models/pascal_voc/ResNet-101/rfcn_end2end/test_agnostic.prototxt',
                            help='the pb file.')

    arg_parser.add_argument('--caffemodel', default='output/resnet101_rfcn_final.caffemodel', help='the label_map file')
    arg_parser.add_argument('-i', '--image', default='images', help='image dir or image file.')
    arg_parser.add_argument('-v', '--video', help='video file to predict.')

    return arg_parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    proto_file_ = os.path.join(base_dir, args.proto_f)
    caffemodel_file_ = os.path.join(base_dir, args.caffemodel)
    images = args.image
    video = args.video

    if os.path.exists(proto_file_) and os.path.exists(caffemodel_file_):
        detector = RFCNDetector(proto_file_, caffemodel_file_)
        if video:
            print('# Predict on videos: {}'.format(video))
            detector.detect_on_video(video)
        else:
            if os.path.isfile(images):
                detector.detect_on_image_list([images])
            elif os.path.isdir(images):
                detector.detect_on_image_list([os.path.join(images, i) for i in os.listdir(images) if i.endswith('.jpg')
                                               or i.endswith('.png') or i.endswith('.jpeg')])
    else:
        print('pb file or label_map file not exist.')
        print('please check: ', proto_file_)
        print(caffemodel_file_)