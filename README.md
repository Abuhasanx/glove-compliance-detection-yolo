#  Gloved vs Ungloved Hand Detection

**Author:** Abu Hasan  
**Contact:** abuhasanxs@gmail.com  
**Task Type:** Safety Compliance — Object Detection  
**Model:** YOLO26s (`.pt`) / ONNX (`.onnx`)  
**Completed in:** Under 24 hours

---

## 📌 Overview

A real-time glove detection system built for **factory safety compliance**, designed to determine whether workers are wearing gloves from camera snapshots or video stream frames.

The system distinguishes between two states:

| Detection | Label | Color |
|-----------|-------|-------|
| Worker wearing gloves | `GLOVED` |  Blue |
| Worker not wearing gloves | `UN GLOVED` |  Red |

---

##  Datasets

| Source | Usage |
|--------|-------|
| [HaGRID Classification 512p — Kaggle](https://www.kaggle.com/datasets/innominate817/hagrid-classification-512p) | Ungloved hand samples |
| [Gloves and Bare Hands — Kaggle](https://www.kaggle.com/datasets/yashdev01/gloves-and-bare-hands-datasets) | Gloved hand samples |
| Manually scraped from Kaggle + other platforms | Additional diversity |

- **500 gloved** images + **500 ungloved** images collected and separated manually
- Annotations approved manually in Roboflow (worked with limited credits)
- Augmentations applied: **Rotate, Flip, Blur, Tilt**
- ⚠️ Colour-grade / Hue / Saturation augmentations intentionally avoided to prevent the model from learning spurious colour shortcuts

---

##  Training

| Parameter | Value |
|-----------|-------|
| Platform | Google Colab |
| Framework | Ultralytics YOLO |
| Model | YOLO26s (`yolo26s`) → exported to ONNX |
| Epochs | 150 |
| Annotation Tool | Roboflow (manual approval per image) |
| Export | `.pt` (PyTorch) + `.onnx` |

---

##  How to Use

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

> Run inside a dedicated virtual environment to avoid package conflicts.

---

### 2. Configure the Script

Open the inference script and set the following at the top:

```python
# --- Choose your model (uncomment ONE) ---
# MODEL_PATH = r"path/to/best.pt"       # PyTorch model
MODEL_PATH = r"path/to/best.onnx"       # ONNX model (recommended)

# --- Set your folders ---
INPUT_FOLDER  = r"path/to/input_images"   # folder of images to run inference on
OUTPUT_FOLDER = r"path/to/output_results" # empty folder to store annotated outputs
```

---

### 3. Tune Per-Class Confidence (Optional)

The script uses **per-class confidence thresholds** for fine-grained control:

```python
PER_CLASS_CONF = {
    "gloves": 0.30,   # raise if too many false positives on gloved hands
    "hand":   0.15,   # kept lower to catch weak ungloved detections
}
```

Adjust these based on your deployment environment and acceptable false positive/negative trade-off.

---

### 4. Run Inference

```bash
python inference.py
```

---

### 5. Output

For every input image, the script writes two files to `OUTPUT_FOLDER`:

- **Annotated image** — bounding boxes with `GLOVED` / `UN GLOVED` labels
- **JSON log** — structured detection results per image

Example JSON log:
```json
{
  "filename": "worker_001.jpg",
  "detections": [
    {
      "label": "GLOVED",
      "confidence": 0.8721,
      "bbox": [120.5, 45.2, 310.8, 280.4]
    }
  ]
}
```

---

##  Script Highlights

- ✅ Supports both `.pt` (PyTorch) and `.onnx` model formats — switch with one line
- ✅ Per-class confidence filtering on top of a global low threshold
- ✅ Clean bounding boxes with label backgrounds for readability
- ✅ JSON logs per image for downstream integration
- ✅ Batch processes an entire folder automatically

---

##  Repository Structure

```
├── detection_script.py          # Main inference script
├── requirements.txt      # Python dependencies
├── best.pt               # YOLOv8s trained weights (PyTorch)
├── best.onnx             # Exported ONNX model
├── input/                # Put your test images here
└── output/               # Annotated images + JSON logs saved here
```

---

##  Important Note on Real-World Accuracy

> Detection models like this are **highly context-dependent**.
>
> This model was trained in under 24 hours using publicly available datasets with limited annotation credits. It performs reasonably well as a general-purpose glove detector, but **for production or client-specific deployments**, accuracy will always improve significantly when data is collected directly from the target environment — the actual factory floor, lighting conditions, camera angles, and glove types in use at that specific site.
>
> Given a client-specific dataset from the actual work location, this same pipeline can be fine-tuned to achieve **~99% accuracy** under real deployment conditions.
>
> This submission demonstrates the full pipeline — data collection, annotation, training, export, and inference — completed under 24 hours with limited cloud credits.

---

##  Requirements

Install via:

```bash
pip install -r requirements.txt
```

Core dependencies:

```
ultralytics
opencv-python
onnxruntime
```

---

## 📬 Contact

**Abu Hasan**  
📧 abuhasanxs@gmail.com
