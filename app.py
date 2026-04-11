import numpy as np
import gradio as gr
import os
import torch
from torch import nn
import torch.nn.functional as F

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# --- Model Definition ---
class Model(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, 3,),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units*2, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=500, out_features=300),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=300, out_features=output_shape),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.classifier(x)
        return x

# --- Setup and Load ---
device = "cuda" if torch.cuda.is_available() else "cpu"

model = Model(
    input_shape=1,
    hidden_units=10,
    output_shape=10
).to(device=device)


if os.path.exists('model_weights.pth'):
    model.load_state_dict(torch.load('model_weights.pth', map_location=device))
model.eval()


# --- Prediction Logic ---
def predict_number(image):
    if image is None :
        return {}, None

    image_rgb = image["composite"][..., :3]
    image_grayScale = np.mean(image_rgb, axis=-1).astype(np.uint8)

    image_grayScale = image_grayScale.reshape(1, 1, 280, 280)
    input_tensor = torch.from_numpy(image_grayScale).float() / 255.0

    input_tensor = F.interpolate(input_tensor, size=(28, 28), mode='bilinear', align_corners=False)

    view_image = (input_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)

    with torch.inference_mode():
        logits = model(input_tensor.to(device))
        probabilities = torch.softmax(logits, dim=1)
        preds = probabilities.cpu().numpy().tolist()[0]

    label_results = {str(i): float(preds[i]) for i in range(10)}

    return label_results, view_image


# --- Gradio UI ---
css = """
.fixed-height img {
    object-fit: contain !important;
    image-rendering: pixelated;
    image-rendering: crisp-edges;
}
"""

with gr.Blocks() as demo:
    gr.Markdown("""
    <center>
    <h1>Digit Classification</h1>
    </center>

    ##  How to use:
    1. **Draw a single digit** (0-9) in the center of the black canvas.
    2. **Click 'Recognize'** to see the prediction and the $28 \times 28$ resized view.
    3. **Click 'Clear canvas'** to start over.
    """)

    with gr.Row():
        with gr.Column():
            sketch_pad = gr.Sketchpad(
                label="Draw here (White on Black)",
                layers=False,
                canvas_size=(280, 280),
                brush=gr.Brush(colors=["#FFFFFF"], default_color="#FFFFFF", default_size=12),
                value={
                    "background": np.zeros((280, 280, 3), dtype=np.uint8),
                    "layers": [],
                    "composite": None
                },
                type="numpy"
            )
            btn = gr.Button("Recognize", variant="primary")
            clear_btn = gr.Button("Clear canvas", variant="stop")

        with gr.Column():
            model_view = gr.Image(
                label="Model Perspective (28x28)",
                image_mode="L",
                container=True,
                height = 280,
                width = 280,
                elem_classes = ["fixed-height"]

            )

            output_label = gr.Label(num_top_classes=3, label="Prediction")

    btn.click(
        fn=predict_number,
        inputs=sketch_pad,
        outputs=[output_label, model_view]
    )

    clear_btn.click(
        fn=lambda: {"background": np.zeros((280, 280, 3), dtype=np.uint8), "layers": [], "composite": None},
        outputs=sketch_pad
    )

    gr.Markdown("""
    ## GitHub Repository
    If you want to see the training process, data augmentation, and how the model was built, check out the **[main_analysis.ipynb](https://github.com/Arcus72/Image-Digit-Classification)** file.""")

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), css=css)