import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.transforms import functional as F


def main():
    model = deeplabv3_resnet50(pretrained=True)
    model.eval()

    # use voc color palette
    voc = Image.open("./imgs/voc_sample.png")
    palette = voc.getpalette()

    @torch.no_grad()
    def inference(gr_input):
        img = Image.fromarray(gr_input.astype("uint8"), "RGB")

        # preprocess
        img = F.to_tensor(img)
        img = img.unsqueeze(0)
        img = F.normalize(
            img,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        # inference
        output = model(img)["out"].squeeze(0)
        _, mask = output.max(dim=0)
        mask = mask.numpy().astype("uint8")

        # convert palette mode image
        mask = Image.fromarray(mask)
        mask = mask.convert("P")
        mask.putpalette(palette)
        mask = mask.convert("RGB")

        return np.asarray(mask)

    inputs = gr.inputs.Image()
    interface = gr.Interface(fn=inference, inputs=inputs, outputs="image")

    interface.launch(server_name="0.0.0.0")


if __name__ == "__main__":
    main()
