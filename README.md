# Microcentrifuge Tube Orientation Detector 

An automated computer vision pipeline that detects microcentrifuge tubes and estimates their orientation angle `[0, 360)` based on the joint-to-tab direction. This project was developed under extreme data constraints (a dataset of only 70 images) and showcases an iterative architectural evolution to solve the 180° bounding-box ambiguity.

##  Final Architecture: CenterNet (Current Best)
**Validation MAE: 32.0°**

After experimenting with Oriented Bounding Boxes (OBB) and continuous angle regression, the final and most successful architecture utilizes **CenterNet**. 

Instead of regressing a single abstract rotation value, CenterNet treats the object detection problem as a keypoint localization task. By accurately identifying the spatial center of the tube and the defining features of the tab, the model successfully mitigates the 180° symmetry ambiguity that plagues standard bounding box models, yielding the lowest Mean Absolute Error (32°) across all tested architectures.

---

## The Engineering Journey: Iterative Models

A core focus of this project was systematically testing different computer vision paradigms against a mathematically complex, low-data environment.

### Iteration 1: YOLOv8-OBB (Baseline)
* **Validation MAE:** 39.57° (Symmetric)
* **Approach:** Utilized YOLOv8n-OBB with heavy online geometric and photometric augmentation (Mosaic, HSV jitter, 180° rotations).
* **Analysis:** The model achieved exceptional detection accuracy (Precision: 0.999, Recall: 1.000). However, because an Oriented Bounding Box is inherently symmetrical, the model could only resolve orientation between 0° and 180°, artificially capping the angle accuracy. 

### Iteration 2: YOLO Region Proposal + ResNet-18 Regression
* **Validation MAE:** 41.0°
* **Approach:** A two-stage pipeline using YOLOv8 to crop the tubes, feeding those standardized crops into a custom-head ResNet-18 optimized via a Cosine LR schedule to explicitly regress the continuous 0-360° joint-to-tab angle.
* **Analysis:** While architecturally sound, this approach struggled due to extreme data starvation (~300 cropped tube instances) and the circular loss trap (standard L1/MSE loss heavily penalizing the boundary between 359° and 1°).

### Iteration 3: CenterNet Keypoint Localization
* **Validation MAE:** 32.0°
* **Approach:** Shifted from angle regression to spatial point detection, allowing the model to focus on the physical location of the tab relative to the tube center.
* **Analysis:** Bypassed the circular loss boundary and the OBB symmetry issues entirely, resulting in the most accurate and stable 360° orientation predictions.

---

## Output Examples
![Annotated Tube Predictions](assets/your_image_name.png)

## Future Work: Overcoming the Data Bottleneck
While the CenterNet architecture successfully reduced the angle error to 32°, the absolute ceiling for accuracy is currently dictated by the 70-image dataset constraint. 

The proposed next step is to build an automated **Synthetic Data Generation Pipeline**. By importing a 3D CAD model of a microcentrifuge tube into a rendering engine (e.g., Blender) via a Python script, we can programmatically drop thousands of tubes onto varying digital surfaces with randomized lighting. This would automatically yield a mathematically perfect, 10,000+ image dataset with exact 360° ground truth orientations, requiring zero manual annotation.

## 📂 Repository Structure
* `centernet_pipeline.ipynb`: The final, best-performing pipeline utilizing CenterNet.
* `yolov8_baseline.ipynb`: The initial YOLOv8-OBB detection and OpenCV annotation pipeline.
* `final_predictions.csv`: The extracted center coordinates, bounding boxes, and angles from the best model.
