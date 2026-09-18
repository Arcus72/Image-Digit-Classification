import numpy as np
import gradio as gr
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import load_model
model = load_model('saved_model.keras')

def predict_number(image):

    image_rgb = image["composite"][..., :3] # delete alpha
    image_grayScale= np.mean(image_rgb, axis=-1).astype(np.uint8)

    image_grayScale = image_grayScale.reshape(1, 280, 280, 1)
    preds = model.predict(image_grayScale)[0]

    return {str(i): float(preds[i]) for i in range(10)}

with gr.Blocks() as demo:
    gr.Markdown("""
    <center>
    <h1>Digit Classification</h1>
    </center>
    
    ##  How to use:
    1. **Draw a single digit** (0-9) in the center of the black canvas.  Digit should be as large as possible.
    2. **Click 'Recognize'** to see the prediction and the model's confidence levels.
    3. **Click 'Clear canvas'** to clear the canvas.
    
    """)
    with gr.Row():
        with gr.Column():
            sketch_pad = gr.Sketchpad(
                label="Draw here (White on Black)",
                layers=False,
                canvas_size=(280, 280),
                brush=gr.Brush(colors=["#FFFFFF"], default_color="#FFFFFF", default_size=10),
                value={
                    "background": np.zeros((280, 280, 3), dtype=np.uint8),
                    "layers": [],
                    "composite": None
                },
                type="numpy"
            )
        with gr.Column():
            output = gr.Label(num_top_classes=3, label="Prediction")

    btn = gr.Button("Recognize", variant="primary")
    clear_btn = gr.Button("Clear canvas", variant="stop")


    clear_btn.click(lambda: {"background": np.zeros((280, 280, 3), dtype=np.uint8), "layers": [], "composite": None},
                    outputs=sketch_pad)
    btn.click(predict_number, inputs=sketch_pad, outputs=output)

    gr.Markdown("""
    ## GitHub Repository
    If you want to see the training process, data augmentation, and how the model was built, check out the **[main_analysis.ipynb](https://github.com/Arcus72/Image-Digit-Classification)** file in my GitHub repository.""")

demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())