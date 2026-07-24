# work by:
# mail : abuhasanxs@gmail.com
# Part 1: Gloved vs Ungloved Hand Detection(Practical Task)Scenario:
# a safety compliance system that checks whether workers are wearing gloves. 
# can deployed on video streams or snapshots from factory cameras

# Dataset : https://www.kaggle.com/datasets/innominate817/hagrid-classification-512p
# Dataset : https://www.kaggle.com/datasets/yashdev01/gloves-and-bare-hands-datasets
# NOTE : DID THIS WORK UNDER 24 HRS OF TIME
# GLOVED AND UNGLOVED DATA HAS BEEN COLLECTED MANUALLY , 500 UNGLOVED AND 500 GLOVED WITH AUGMENTATION
# YOLO26s AND ONNX MODEL


#######################################################################################################
import sys
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

# ----------------------------------------------------------------------
# MODEL SELECTION - uncomment ONE of the two lines below
# ----------------------------------------------------------------------
# MODEL_PATH = r"D:\vision\GLOVE TEST FILES\best.pt"          # PyTorch model
MODEL_PATH = r"D:\vision\GLOVE TEST FILES\best.onnx"        # ONNX model (needs: pip install onnxruntime)

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
INPUT_FOLDER  = r"D:\vision\images to test\input"
OUTPUT_FOLDER = r"D:\vision\images to test\output"

# Global "capture" threshold passed to the model - kept LOW so weak "hand"
# detections aren't discarded before the per-class filter below gets a look.


################################################################################################

#AS YOU CAN NOTICE I GAVE PER CLASS CONFIDENCE

CONF_THRESH   = 0.1
IOU_THRESH    = 0.5
IMGSZ         = 640
IMAGE_EXTS    = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

PER_CLASS_CONF = {
    "gloves": 0.30,
    "glove":  0.30,
    "hand":   0.15,   
}

################################################################################################


# Test-time augmentation (multi-scale/flip inference, merged) - can help
# recall on a weak class. EXPERIMENTAL for YOLO26's newer head, .pt only.
USE_TTA = False

# --- display styling ---
BOX_THICKNESS   = 2               
FONT            = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE      = 0.5
FONT_THICKNESS  = 1
COLOR_GLOVE     = (255, 0, 0) 
COLOR_HAND      = (0, 0, 255)     
COLOR_DEFAULT   = (0, 255, 0)      
COLOR_LABEL_BG  = (0, 0, 0)        
COLOR_LABEL_TXT = (255, 255, 255)  


LABEL_STYLE = {
    "gloves": ("GLOVED", COLOR_GLOVE),
    "glove":  ("GLOVED", COLOR_GLOVE),
    "hand":   ("UN GLOVED", COLOR_HAND),
}


def style_for(raw_class_name: str):
    key = raw_class_name.lower()
    if key in LABEL_STYLE:
        return LABEL_STYLE[key]
    return raw_class_name.upper(), COLOR_DEFAULT  # unknown class fallback


def draw_detection(img, x1, y1, x2, y2, label_text, color):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # thin, sharp bounding box (no anti-aliasing = crisp edges)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, BOX_THICKNESS, lineType=cv2.LINE_8)

    (text_w, text_h), baseline = cv2.getTextSize(label_text, FONT, FONT_SCALE, FONT_THICKNESS)
    pad = 4

    
    if y1 - text_h - baseline - pad < 0:
        bg_top = y1
        bg_bottom = y1 + text_h + baseline + pad
    else:
        bg_top = y1 - text_h - baseline - pad
        bg_bottom = y1

    cv2.rectangle(img, (x1, bg_top), (x1 + text_w + 2 * pad, bg_bottom), COLOR_LABEL_BG, -1)
    text_y = bg_bottom - baseline - 2
    cv2.putText(img, label_text, (x1 + pad, text_y), FONT, FONT_SCALE,
                COLOR_LABEL_TXT, FONT_THICKNESS, cv2.LINE_AA)


def main():
    if not INPUT_FOLDER or not OUTPUT_FOLDER:
        sys.exit("ERROR: set INPUT_FOLDER and OUTPUT_FOLDER at the top of the script.")

    input_dir = Path(INPUT_FOLDER)
    output_dir = Path(OUTPUT_FOLDER)

    if not input_dir.exists():
        sys.exit(f"ERROR: input folder not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not Path(MODEL_PATH).exists():
        sys.exit(f"ERROR: model not found at {MODEL_PATH}")

    is_onnx = Path(MODEL_PATH).suffix.lower() == ".onnx"
    model_kind = "ONNX" if is_onnx else "PyTorch (.pt)"
    print(f"Using {model_kind} model: {MODEL_PATH}")

    use_tta = USE_TTA and not is_onnx
    if USE_TTA and is_onnx:
        print("NOTE: USE_TTA is on but this is an ONNX model - TTA isn't supported "
              "on exported models, so running without it. Use the .pt model to test TTA.")

    image_paths = sorted(
        [p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    )
    if not image_paths:
        sys.exit(f"ERROR: no images found in {input_dir} (looked for {IMAGE_EXTS})")

    print(f"Found {len(image_paths)} images in {input_dir}")
    model = YOLO(MODEL_PATH)
    class_names = model.names  # raw names as trained, e.g. {0: 'gloves', 1: 'hand'}

    total_detections = 0
    total_dropped_by_class_conf = 0

    for idx, img_path in enumerate(image_paths, 1):
        results = model.predict(
            source=str(img_path),
            conf=CONF_THRESH,
            iou=IOU_THRESH,
            imgsz=IMGSZ,
            augment=use_tta,
            verbose=False,
        )
        result = results[0]

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  WARNING: could not read {img_path.name}, skipping.")
            continue

        detections = []
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                raw_name = class_names[cls_id]

                # per-class confidence floor - the real filtering step
                floor = PER_CLASS_CONF.get(raw_name.lower(), CONF_THRESH)
                if conf < floor:
                    total_dropped_by_class_conf += 1
                    continue

                x1, y1, x2, y2 = [round(v, 2) for v in box.xyxy[0].tolist()]
                display_label, color = style_for(raw_name)
                draw_detection(img, x1, y1, x2, y2, display_label, color)

                detections.append({
                    "label": display_label,
                    "confidence": round(conf, 4),
                    "bbox": [x1, y1, x2, y2],
                })

        total_detections += len(detections)

        out_img_path = output_dir / img_path.name
        cv2.imwrite(str(out_img_path), img)

        log = {
            "filename": img_path.name,
            "detections": detections,
        }
        json_path = output_dir / f"{img_path.stem}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)

        print(f"[{idx}/{len(image_paths)}] {img_path.name}: {len(detections)} detection(s)")

    print(f"\nDone. {total_detections} total detections across {len(image_paths)} images.")
    print(f"({total_dropped_by_class_conf} candidate boxes were filtered out by PER_CLASS_CONF)")
    print(f"Annotated images + JSON logs saved to: {output_dir}")


if __name__ == "__main__":
    main()