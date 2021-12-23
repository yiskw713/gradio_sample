# gradio samples

This repo contains gradio demo codes of:

* visualization of image preprocessing
* image classification with ResNet50
* semantic segmentation with DeepLabV3
* object detection with YOLOX

All applications can be accessible on `localhost:7860`.

## Usage

```sh
cd <PROJECT_ROOT>

# download yolox pre-trained weight
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx \
    -O weights/yolox_s.onnx

# build docker image
sh scripts/build_docker.sh

# run docker container
sh scripts/run_docker.sh
```

in docker environment

```sh
# visualize preprocess
python src/visualize_preprocess.py

# image classification
python src/image_classification.py

# semantic segmentation
python src/semantic_segmentation.py

# object detection
python src/object_detection.py
```

## Reference

* [gradio-app/gradio: Create UIs for your machine learning model in Python in 3 minutes](https://github.com/gradio-app/gradio)
* [torchvision.models — Torchvision 0.11.0 documentation](https://pytorch.org/vision/stable/models.html)
* [Megvii-BaseDetection/YOLOX: YOLOX is a high-performance anchor-free YOLO, exceeding yolov3~v5 with MegEngine, ONNX, TensorRT, ncnn, and OpenVINO supported. Documentation: https://yolox.readthedocs.io/](https://github.com/Megvii-BaseDetection/YOLOX)
