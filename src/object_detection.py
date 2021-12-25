import cv2
import gradio as gr
import numpy as np
import onnxruntime

from yolox_utils import COCO_CLASSES, demo_postprocess, multiclass_nms
from yolox_utils import preproc as preprocess
from yolox_utils import vis


def main():
    MODEL = "./weights/yolox_s.onnx"
    INPUT_SHAPE = "640,640"
    WITH_P6 = False

    input_shape = tuple(map(int, INPUT_SHAPE.split(",")))
    session = onnxruntime.InferenceSession(MODEL)

    def inference(gr_input, score_thr, nms_iou_thr):
        img, ratio = preprocess(gr_input, input_shape)

        ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
        output = session.run(None, ort_inputs)
        predictions = demo_postprocess(output[0], input_shape, p6=WITH_P6)[0]

        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=nms_iou_thr, score_thr=0.01)
        if dets is not None:
            final_boxes, final_scores, final_cls_inds = (
                dets[:, :4],
                dets[:, 4],
                dets[:, 5],
            )
            gr_input = vis(
                gr_input,
                final_boxes,
                final_scores,
                final_cls_inds,
                conf=score_thr,
                class_names=COCO_CLASSES,
            )

        return np.asarray(gr_input)

    img_input = gr.inputs.Image()
    score_thr_input = gr.inputs.Slider(
        minimum=0,
        maximum=1.0,
        step=0.05,
        default=0.3,
        label="score threshold",
    )
    nms_iou_thr_input = gr.inputs.Slider(
        minimum=0,
        maximum=1.0,
        step=0.05,
        default=0.45,
        label="nms iou threshold",
    )

    interface = gr.Interface(
        fn=inference,
        inputs=[
            img_input,
            score_thr_input,
            nms_iou_thr_input,
        ],
        outputs="image",
    )

    interface.launch(server_name="0.0.0.0")


if __name__ == "__main__":
    main()
