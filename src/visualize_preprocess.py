import gradio as gr
from PIL import Image
from torchvision import transforms
import numpy as np


def preprocess(img: np.ndarray, operation: str) -> np.ndarray:
    img = Image.fromarray(img.astype("uint8"), "RGB")

    if operation == "color_jitter":
        color_jitter = transforms.ColorJitter(
            brightness=0.5,
            contrast=0.5,
            saturation=0.5,
            hue=0.5,
        )
        img = color_jitter(img)
    elif operation == "affine":
        random_affine = transforms.RandomAffine(
            degrees=15,
            shear=20,
        )
        img = random_affine(img)
    elif operation == "resize_crop":
        resize_crop = transforms.RandomResizeCrop(
            size=(224, 224),
            ratio=(3 / 4, 4 / 3),
        )
        img = resize_crop(img)
    elif operation == "perspective":
        perspective = transforms.RandomPerspective(distortion_scale=0.2, p=1.0)
        img = perspective(img)
    else:
        raise ValueError

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

    interface.launch(debug=True, share=False)


if __name__ == "__main__":
    main()
