# gradio samples

This repo contains sample codes of

* visualization of image preprocessing
* image classification with ResNet50
* semantic segmentation with DeepLabV3


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

# in docker environment
# image classification
python src/image_classification.py
```

## Reference

[gradio-app/gradio: Create UIs for your machine learning model in Python in 3 minutes](https://github.com/gradio-app/gradio)
