
# Data Source Documentation
## Pothole Priority Predictor

## Dataset Overview
- **Name:** Pothole Detection Dataset
- **Source:** Roboflow Universe
- **URL:** https://universe.roboflow.com/major-vl1h9/pothole-bwzav/dataset/2
- **Format:** YOLOv11
- **License:** CC BY 4.0

## Dataset Statistics
| Split      | Images |
|------------|--------|
| Train      | 6809   |
| Validation | 1944   |
| Test       | 974    |
| **Total**  | **9727** |

## Classes
| Class ID | Class Name | Annotations |
|----------|------------|-------------|
| 0        | Pothole    | 4829        |

## Preprocessing
- Image size: 640x640
- Augmentations: Blur, MedianBlur, ToGray, Mosaic
- Annotation format: YOLO (x_center, y_center, width, height)

## Model Trained On This Dataset
- Model: YOLOv11n
- Epochs: 50
- Batch Size: 16
- mAP50: 0.883
- Precision: 0.861
- Recall: 0.810
