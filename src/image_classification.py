import gradio as gr
import requests
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import resnet50
from torchvision.transforms import functional as F


def main():
    model = resnet50(pretrained=True)
    model.eval()

    # get ImageNet lables
    response = requests.get("https://git.io/JJkYN")
    labels = response.text.split("\n")

    @torch.no_grad()
    def inference(gr_input):
        img = Image.fromarray(gr_input.astype("uint8"), "RGB")

        # preprocess
        img = F.resize(img, (224, 224))
        img = F.to_tensor(img)
        img = img.unsqueeze(0)
        img = F.normalize(
            img,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        # inference
        output = model(img).squeeze(0)
        probs = nn.functional.softmax(output, dim=0).numpy()

        return {labels[i]: float(probs[i]) for i in range(1000)}

    inputs = gr.inputs.Image()
    outputs = gr.outputs.Label(num_top_classes=5)
    interface = gr.Interface(fn=inference, inputs=inputs, outputs=outputs)

    interface.launch(server_name="0.0.0.0")


if __name__ == "__main__":
    main()
