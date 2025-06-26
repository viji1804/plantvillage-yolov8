import gradio as gr
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load trained YOLOv8 model
model = YOLO("best.pt")

# Prediction function
def detect_objects(image):
    if image is None:
        return None, "⚠️ Please upload a valid plant leaf image."

    # Run prediction
    results = model.predict(source=image, conf=0.25)
    if len(results[0].boxes) == 0:
        return image, "✅ No disease detected – the leaf appears healthy!"

    # Annotate and return
    result_image = results[0].plot()
    return Image.fromarray(result_image), "✅ Disease detected and highlighted."

# Gradio UI with better layout
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🌿 PlantVillage Disease Detection
        Upload an image of a **plant leaf** to detect diseases using a YOLOv8 model.
        """
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="📷 Upload Plant Leaf Image", tool="editor")
            submit_btn = gr.Button("🔍 Detect Disease")

        with gr.Column():
            output_image = gr.Image(type="pil", label="🩺 Detection Result")
            feedback = gr.Textbox(label="Result Message", interactive=False)

    submit_btn.click(fn=detect_objects, inputs=input_image, outputs=[output_image, feedback])

# Run the app
if __name__ == "__main__":
    demo.launch()
