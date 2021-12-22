import gradio as gr
import torch
import torch.nn as nn
from torchvision.transforms import functional as F
from torchvision.models import resnet50
import requests
from PIL import Image


import argparse
import os

import cv2
import numpy as np

import onnxruntime

from yolox_utils import preproc as preprocess
from yolox_utils import COCO_CLASSES, multiclass_nms, demo_postprocess, vis


if __name__ == "__main__":
    MODEL = "./weights/yolox_s.onnx"
    SCORE_TH = 0.3
    INPUT_SHAPE = "640,640"
    WITH_P6 = False

    input_shape = tuple(map(int, INPUT_SHAPE.split(",")))
    session = onnxruntime.InferenceSession(MODEL)

    # TODO: opencv install
    def inference(gr_input):
        # RGB2BGR?
        img, ratio = preprocess(origin_img, input_shape)

        ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
        output = session.run(None, ort_inputs)
        predictions = demo_postprocess(output[0], input_shape, p6=args.with_p6)[0]

        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
        if dets is not None:
            final_boxes, final_scores, final_cls_inds = (
                dets[:, :4],
                dets[:, 4],
                dets[:, 5],
            )
            origin_img = vis(
                origin_img,
                final_boxes,
                final_scores,
                final_cls_inds,
                conf=SCORE_TH,
                class_names=COCO_CLASSES,
            )

        return np.asarray(original_img)
