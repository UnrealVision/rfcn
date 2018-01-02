"""
this file contains a tool kit for computer vision kit
we can using this powerful kit to display detection or
segmentation skillfully
"""
import numpy as np
import colorsys
import cv2
import time
import os


def combine_detections(classes, scores, boxes):
    """
    combines those into one single detection format
    :param boxes:
    :param scores:
    :param classes:
    :return: detections: ssd detections, numpy.array([[id, score, x1, y1, x2, y2]...])
    """
    if isinstance(boxes, list):
        boxes = np.array(boxes)
    if isinstance(scores, list):
        scores = np.array(scores)
    if isinstance(classes, list):
        classes = np.array(classes)

    if boxes.shape[0] != scores.shape[0] != classes.shape[0]:
        print('boxes and scores and classes are not the same samples, can not combine into detections.')
        exit(0)
    else:
        # classes and score maybe (100, 1) or (1, 100)
        classes = np.squeeze(classes)
        scores = np.squeeze(scores)
        boxes = np.squeeze(boxes)
        try:
            tmp = np.stack([classes, scores], axis=1)
            detections = np.hstack([tmp, boxes])
            return detections
        except Exception as e:
            print(e)
            print('can not combine detections, the shapes are: ')
            print('classes: ', classes.shape)
            print('scores: ', scores.shape)
            print('boxes: ', boxes.shape)
            print('make sure they are same at the first dimension.')
            exit(0)


def visualize_det_cv2(img, detections, classes=None, thresh=0.6, is_show=False, background_id=-1,
                      is_normalized=False, ignore_zero=False, is_video=False):
    """
    visualize detection on image using cv2, this is the standard way to visualize detections
    :param img:
    :param detections: ssd detections, numpy.array([[id, score, x1, y1, x2, y2]...])
            each row is one object
    :param classes:
    :param thresh:
    :param is_show:
    :param background_id: -1
    :param is_normalized
    :param ignore_zero
    :param is_video
    :return:
    """
    assert classes, 'from visualize_det_cv2, classes must be provided, each class in a list with' \
                    ' certain order.'
    assert isinstance(img, np.ndarray), 'from visualize_det_cv2, img must be a numpy array object.'

    height = img.shape[0]
    width = img.shape[1]

    font = cv2.QT_FONT_NORMAL
    font_scale = 0.4
    font_thickness = 1
    line_thickness = 1

    for i in range(detections.shape[0]):
        cls_id = int(detections[i, 0])
        if cls_id != background_id:
            if not ignore_zero:
                cls_id -= 1
            score = detections[i, 1]
            if score > thresh:
                unique_color = _create_unique_color_uchar(cls_id)

                # if detection coordinates normalized, then do this step, otherwise not
                # y_min, x_min, y_max, x_max
                if is_normalized:
                    y1 = int(detections[i, 2] * height)
                    x1 = int(detections[i, 3] * width)
                    y2 = int(detections[i, 4] * height)
                    x2 = int(detections[i, 5] * width)
                else:
                    y1 = int(detections[i, 2])
                    x1 = int(detections[i, 3])
                    y2 = int(detections[i, 4])
                    x2 = int(detections[i, 5])

                cv2.rectangle(img, (x1, y1), (x2, y2), unique_color, line_thickness)

                try:
                    text_label = '{} {:.2f}'.format(classes[cls_id], score)
                except IndexError as e:
                    print('e')
                    text_label = '{} {:.2f}'.format('unknown class', score)
                (ret_val, base_line) = cv2.getTextSize(text_label, font, font_scale, font_thickness)
                text_org = (x1, y1 - 0)

                cv2.rectangle(img, (text_org[0] - 5, text_org[1] + base_line + 2),
                              (text_org[0] + ret_val[0] + 5, text_org[1] - ret_val[1] - 2), unique_color, line_thickness)
                # this rectangle for fill text rect
                cv2.rectangle(img, (text_org[0] - 5, text_org[1] + base_line + 2),
                              (text_org[0] + ret_val[0] + 4, text_org[1] - ret_val[1] - 2),
                              unique_color, -1)
                cv2.putText(img, text_label, text_org, font, font_scale, (255, 255, 255), font_thickness)
    if is_video:
        cv2.imshow('video', img)
        cv2.waitKey(1)
    else:
        if is_show:
            cv2.imshow('image', img)
            cv2.waitKey(0)
    return img


def visualize_lane_cv2(img):
    # visualize lane detect result on an image
    pass


def visualize_det_mask_cv2(img, detections, masks, classes=None, is_show=False, background_id=-1, is_video=False):
    """
    this method using for display detections and masks on image
    :param img:
    :param detections: numpy.array([[id, score, x1, y1, x2, y2]...])
            each row is one object. contains id and score in the first 2 rows
    :param masks: numpy.array([[mask_width, mask_height], ...], every element is an
    one chanel mask of on object
    :param classes: classes names in a list with certain order
    :param is_show: to show if it is video
    :param background_id
    :param is_video
    :return:
    """
    assert isinstance(img, np.ndarray) and isinstance(detections, np.ndarray) and isinstance(masks, np.ndarray),\
        'images and detections and masks must be numpy array'
    assert detections.shape[0] == masks.shape[-1], 'detections nums and masks nums are not equal'
    assert is_show != is_video, 'you can not set is_show and is_video at the same time.'
    # draw detections first
    img = visualize_det_cv2(img, detections, classes=classes, is_show=False)

    masked_image = img
    print('masked image shape: ', masked_image.shape)
    num_instances = detections.shape[0]
    for i in range(num_instances):
        cls_id = int(detections[i, 0])
        if cls_id != background_id:
            unique_color = _create_unique_color_uchar(cls_id)
            mask = masks[:, :, i]
            masked_image = _apply_mask2(masked_image, mask, unique_color)
    # masked_image = masked_image.astype(int)
    if is_video:
        cv2.imshow('image', masked_image)
        cv2.waitKey(1)
    elif is_show:
        cv2.imshow('image', masked_image)
        cv2.waitKey(0)
    return masked_image


def _apply_mask2(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = np.where(mask == 1, image[:, :, c] * (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image


def _create_unique_color_float(tag, hue_step=0.41, alpha=0.7):
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b, alpha


def _create_unique_color_uchar(tag, hue_step=0.41, alpha=0.7):
    r, g, b, a = _create_unique_color_float(tag, hue_step, alpha)
    return int(255 * r), int(255 * g), int(255 * b), int(255 * a)
