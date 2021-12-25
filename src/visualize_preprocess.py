import gradio as gr
import numpy as np
from PIL import Image
from torchvision import transforms


def preprocess(img: np.ndarray, operation: str) -> np.ndarray:
    img = Image.fromarray(img.astype("uint8"), "RGB")

    if operation == "color_jitter":
        preprocess = transforms.ColorJitter(
            brightness=0.5,
            contrast=0.5,
            saturation=0.5,
            hue=0.5,
        )
    elif operation == "affine":
        preprocess = transforms.RandomAffine(
            degrees=15,
            shear=20,
        )
    elif operation == "resize_crop":
        preprocess = transforms.RandomResizedCrop(
            size=(224, 224),
            ratio=(3 / 4, 4 / 3),
        )
    elif operation == "perspective":
        preprocess = transforms.RandomPerspective(distortion_scale=0.2, p=1.0)
    else:
        raise ValueError("Invalid preprocess type.")

    img = preprocess(img)
    return np.asarray(img)


def main():
    gr_img = gr.inputs.Image()

    interface = gr.Interface(
        fn=preprocess,
        inputs=[
            gr_img,
            gr.inputs.Radio(
                [
                    "color_jitter",
                    "affine",
                    "resize_crop",
                    "perspective",
                ]
            ),
        ],
        outputs="image",
        title="",
        description="drop image and choose pre-process",
    )

    interface.launch(server_name="0.0.0.0")


if __name__ == "__main__":
    main()
