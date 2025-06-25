
# PlantVillage Disease Detection using YOLOv8

A deep learning-based web application for **real-time plant disease detection** using the **YOLOv8 object detection model**, trained on the PlantVillage dataset.

[![Hugging Face Spaces](https://img.shields.io/badge/Live%20Demo-On%20Hugging%20Face-blue?logo=huggingface)](https://viji1804-plantvillage-yolov8.hf.space)

---

##  Project Overview

Plant diseases can severely affect crop yield and quality. This project leverages **YOLOv8**, a state-of-the-art object detection model from **Ultralytics**, to classify and localize plant leaf diseases from images using the **PlantVillage dataset**.

It is deployed on [Hugging Face Spaces](https://huggingface.co/spaces/viji1804/plantvillage-yolov8) with **Gradio** for easy and interactive usage.

---

##  Try the Live Demo

[**Click here to use the model**](https://viji1804-plantvillage-yolov8.hf.space)

Upload any leaf image and get instant predictions with bounding boxes and labels using the trained YOLOv8 model.

---

## 💡 Features

- 🧠 Trained on the PlantVillage Dataset
- ⚡️ Real-time object detection with YOLOv8
- 🖼️ Upload your own image to test
- 📊 High accuracy and performance
- 🌍 Deployed on Hugging Face using Gradio

---

## 📁 Project Structure

```
plantvillage-yolov8/
├── app.py               # Gradio interface
├── best.pt              # Trained YOLOv8 model
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
```

---

## ⚙️ Run Locally

```bash
git clone https://github.com/viji1804/plantvillage-yolov8.git
cd plantvillage-yolov8
pip install -r requirements.txt
python app.py
```

---



## 📦 Requirements

```
ultralytics
gradio
Pillow
```

---

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [PlantVillage Dataset](https://www.kaggle.com/datasets/smaranjitghose/plantvillage-dataset)
- [Gradio](https://gradio.app/)
- [Hugging Face Spaces](https://huggingface.co/spaces)

---

## Author

**Vijayalakshmi S**  
[GitHub: @viji1804](https://github.com/viji1804)
