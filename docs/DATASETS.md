# Dataset Preparation Guide

This guide covers dataset collection, annotation, and preparation for training YOLO models for DeepStream deployment.

---

## 📋 Dataset Requirements

### General Requirements
- **Format**: YOLO format (one `.txt` file per image)
- **Minimum images**: 1000+ for good performance
- **Train/Val/Test split**: 70/20/10 recommended
- **Image quality**: Minimum 640px shortest side
- **Diversity**: Various conditions (lighting, weather, angles, distances)

---

## 🎯 Domain-Specific Datasets

### PPE Detection

**Classes**: person, helmet, no_helmet, vest, no_vest, harness

**Public Datasets**:
- [Roboflow PPE Detection](https://universe.roboflow.com) - Search "PPE"
- [Kaggle Safety Helmet Dataset](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection)
- [Construction Helmet and Vest (CHV)](https://github.com/KevinMusgrave/pytorch-metric-learning)

**Collection Tips**:
- Include various site conditions (construction, mining, manufacturing)
- Day and night lighting
- Different distances (close-up to far away)
- Partial occlusions (people behind objects)
- Different helmet/vest colors

---

### Traffic Monitoring

**Classes**: person, bicycle, car, motorcycle, bus, truck, traffic_light, stop_sign

**Public Datasets**:
- [BDD100K](https://www.bdd100k.com/) - Large-scale driving dataset
- [COCO](https://cocodataset.org/) - Subset with vehicles
- [NVIDIA TrafficCamNet](https://catalog.ngc.nvidia.com/models) - Pre-trained, can fine-tune

**Collection Tips**:
- Multiple camera angles (overhead, side-view, dash-cam)
- Rush hour vs normal traffic
- Day/night/dawn/dusk
- Clear and rainy weather
- Urban and highway scenarios

---

### Wildlife Detection

**Classes**: Depends on your target species

**Public Datasets**:
- [Microsoft MegaDetector](https://github.com/microsoft/CameraTraps)
- [LILA BC](https://lila.science/) - Labeled camera trap images
- [iNaturalist](https://www.inaturalist.org/pages/developers) - Millions of wildlife images
- [Snapshot Serengeti](https://www.snapshotserengeti.org/)

**Collection Tips**:
- Camera trap images at multiple locations
- Various times of day
- Different seasons
- Include false positives (vegetation motion)
- Include humans/vehicles for differentiation

---

### Industrial/Mining Equipment

**Classes**: haul_truck, excavator, bulldozer, wheel_loader, dump_truck, grader, wagon, conveyor, person, light_vehicle

**Public Datasets**:
- [Roboflow Construction](https://universe.roboflow.com) - Search "construction equipment"
- **Custom annotation required** - Most mining-specific equipment

**Collection Tips**:
- Equipment at various scales (close and far)
- Multiple angles
- Day and night operations
- Dusty conditions (common in mining)
- Partial occlusions (equipment behind terrain)
- Include operators/workers for safety detection

---

## 🛠️ Annotation Tools

### Recommended Tools

| Tool | Type | Best For | Cost |
|------|------|----------|------|
| [CVAT](https://cvat.org/) | Self-hosted | Video annotation | Free |
| [Label Studio](https://labelstud.io/) | Self-hosted | Flexible formats | Free |
| [Roboflow](https://roboflow.com/) | Cloud | Managed + augmentation | Free tier + paid |
| [Labelbox](https://labelbox.com/) | Cloud | Enterprise | Paid |

### CVAT Setup (Recommended for Video)

```bash
# Install with Docker
git clone https://github.com/opencv/cvat
cd cvat
docker-compose up -d

# Access at http://localhost:8080
# Default credentials: see CVAT docs
```

**CVAT Workflow**:
1. Create project
2. Upload video or images
3. Define labels (classes)
4. Annotate with bounding boxes
5. Export as YOLO 1.1 format

### Roboflow (Easiest Option)

1. Create account at [roboflow.com](https://roboflow.com)
2. Create project and upload images
3. Annotate or use Smart Polygon
4. Apply augmentations (optional)
5. Generate dataset
6. Export as YOLOv8 format

---

## 📦 Dataset Preparation Workflow

### 1. Collect/Download Raw Data

```bash
# Create directory structure
mkdir -p datasets/raw/{ppe,traffic,wildlife,industrial}

# Extract frames from video (if applicable)
python scripts/prepare_dataset.py extract \
    --video site_footage.mp4 \
    --output datasets/raw/ppe/images \
    --fps 1  # 1 frame per second
```

### 2. Annotate

Use CVAT, Label Studio, or Roboflow to annotate. Export in YOLO format.

Expected structure after annotation:
```
datasets/raw/ppe/
├── images/
│   ├── frame_000001.jpg
│   ├── frame_000002.jpg
│   └── ...
└── labels/
    ├── frame_000001.txt
    ├── frame_000002.txt
    └── ...
```

YOLO label format (one line per object):
```
class_id center_x center_y width height
```

Example (`frame_000001.txt`):
```
0 0.5 0.5 0.3 0.4    # person at center
1 0.52 0.42 0.08 0.1 # helmet on person's head
```

### 3. Split Dataset

```bash
python scripts/prepare_dataset.py split \
    --images datasets/raw/ppe/images \
    --labels datasets/raw/ppe/labels \
    --output datasets/processed/ppe \
    --train 0.7 \
    --val 0.2 \
    --test 0.1
```

Result:
```
datasets/processed/ppe/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### 4. Create Dataset YAML

```bash
python scripts/prepare_dataset.py yaml \
    --dataset-dir datasets/processed/ppe \
    --classes person helmet no_helmet vest no_vest harness \
    --output training/configs/datasets/ppe.yaml
```

Resulting `ppe.yaml`:
```yaml
path: /path/to/datasets/processed/ppe
train: train/images
val: val/images
test: test/images
nc: 6
names:
  0: person
  1: helmet
  2: no_helmet
  3: vest
  4: no_vest
  5: harness
```

---

## 📊 Dataset Quality Checks

### Check Class Distribution

```python
import os
from collections import Counter

labels_dir = "datasets/processed/ppe/train/labels"
class_counts = Counter()

for label_file in os.listdir(labels_dir):
    with open(os.path.join(labels_dir, label_file)) as f:
        for line in f:
            class_id = int(line.split()[0])
            class_counts[class_id] += 1

for class_id, count in sorted(class_counts.items()):
    print(f"Class {class_id}: {count} instances")
```

### Check Image Sizes

```python
from PIL import Image
import os

images_dir = "datasets/processed/ppe/train/images"
sizes = []

for img_file in os.listdir(images_dir):
    img = Image.open(os.path.join(images_dir, img_file))
    sizes.append(img.size)

print(f"Image sizes: {len(set(sizes))} unique")
print(f"Common sizes: {Counter(sizes).most_common(5)}")
```

---

## 🎨 Data Augmentation

YOLO training includes augmentation by default, but for additional augmentation:

### Roboflow Augmentation

Roboflow provides built-in augmentation:
- Brightness/contrast adjustments
- Random crops
- Rotations
- Flips
- Noise injection

### Custom Augmentation

If needed, use libraries like Albumentations:

```python
import albumentations as A

transform = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.5),
    A.GaussNoise(p=0.3),
    A.MotionBlur(p=0.3),
], bbox_params=A.BboxParams(format='yolo'))
```

---

## 📝 Calibration Dataset

For INT8 quantization, create a calibration subset:

```bash
# Extract 500-1000 representative images from training set
mkdir -p datasets/calibration/ppe/images

# Randomly sample
find datasets/processed/ppe/train/images -type f | shuf -n 500 | \
    xargs -I {} cp {} datasets/calibration/ppe/images/

# Create calibration YAML
python scripts/prepare_dataset.py yaml \
    --dataset-dir datasets/calibration/ppe \
    --classes person helmet no_helmet vest no_vest harness \
    --output training/configs/calibration/ppe_calib.yaml
```

---

## ✅ Checklist Before Training

- [ ] Dataset has 1000+ total images
- [ ] Train/val/test split is complete
- [ ] All classes are represented (>100 instances each)
- [ ] Labels are in YOLO format
- [ ] Images are minimum 640px shortest side
- [ ] Dataset YAML is correct
- [ ] Calibration dataset prepared (if using INT8)

---

## 🐛 Common Issues

### Issue: Class Imbalance
**Solution**: Use weighted sampling or augment minority classes

### Issue: Small Objects Not Detected
**Solution**: 
- Use higher image resolution (1280px)
- Use YOLOv8m or YOLOv8l
- Ensure small objects are clearly visible in training data

### Issue: Poor Performance on Specific Condition
**Solution**: Add more training data for that condition (e.g., night scenes)

---

## 📚 Additional Resources

- [Ultralytics YOLO Docs](https://docs.ultralytics.com/)
- [Roboflow Dataset Search](https://universe.roboflow.com/)
- [COCO Dataset Format](https://cocodataset.org/#format-data)
- [CVAT Documentation](https://opencv.github.io/cvat/docs/)
