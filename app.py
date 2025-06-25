import gradio as gr
from ultralytics import YOLO
from PIL import Image

# Load trained model
model = YOLO("best.pt")  

# Define prediction function
def detect_objects(image):
    results = model.predict(source=image, conf=0.25)
    result_image = results[0].plot()
    return Image.fromarray(result_image)

# Gradio UI
demo = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="pil", label="Upload Plant Image"),
    outputs=gr.Image(type="pil", label="Detection Result"),
    title="PlantVillage Disease Detection (YOLOv8)",
    description="Upload an image of a plant leaf to detect its disease using a trained YOLOv8 model.",
)

if __name__ == "__main__":
    demo.launch()
