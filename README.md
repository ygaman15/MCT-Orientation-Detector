# MCT-Orientation-Detector

An automated computer vision pipeline that detects microcentrifuge tubes and estimates their orientation angle using a hybrid YOLOv8 Oriented Bounding Box (OBB) approach.

This project solves the challenge of detecting tubes and calculating their rotation angle `[0, 360)` based on the joint-to-tab direction, utilizing a highly constrained dataset of only 70 images. 

To prevent overfitting, the architecture utilizes **YOLOv8n-OBB** coupled with heavy online geometric and photometric augmentation (Mosaic, HSV jitter, 180° rotations).

## 📊 Performance Metrics (Validation Set)
The model achieved exceptional detection accuracy, successfully navigating varying background colors and lighting conditions:
* Precision:  0.999
* Recall: 1.000
* F1-Score: ~1.00
* Symmetric Angle Error (MAE): 39.57°

*Note: Because an Oriented Bounding Box is inherently symmetrical, the model resolves orientation between 0° and 180°. The Val MAE was calculated symmetrically to evaluate the raw bounding box fit prior to 360° tab-classification.*

#  Output Examples
*(To make this work, replace the link below with the actual name of an image you uploaded to your assets folder)*
![Annotated Tube Predictions](assets/your_image_name.png)

#Future Work & Proposed Architecture (Phase 3)

While the two-stage YOLO-to-ResNet pipeline successfully demonstrated the concept of region-cropping followed by angle regression, the 41° Validation MAE highlights the mathematical limitations of using standard continuous regression on a constrained dataset. 

To achieve true 360° awareness and reduce the angle error, the following architectural upgrades are proposed for Phase 3:

# 1. Shift to 2D Keypoint Detection
CNNs are fundamentally better at predicting spatial coordinates than abstract rotation values. Instead of regressing a single angle, the problem should be reframed as a Pose Estimation task (e.g., using YOLO-Pose). The model will predict exactly two coordinates:
* The absolute center of the tube cap $(x_1, y_1)$
* The tip of the directional plastic tab $(x_2, y_2)$

Once predicted, the orientation can be calculated deterministically with zero machine learning using the arctangent function:
$\theta = \arctan2(y_2 - y_1, x_2 - x_1) \times \left( \frac{180}{\pi} \right)$

# 2. Implement Circular Loss & Trigonometric Embeddings
Standard loss functions (L1/MSE) mathematically break at the 0°/360° boundary (e.g., penalizing the model 358° for predicting 359° when the true angle is 1°). To solve this, the ResNet regression head must use a custom circular loss function:
$L = \min(|\theta_{pred} - \theta_{true}|, 360 - |\theta_{pred} - \theta_{true}|)$

Alternatively, the model can regress two continuous variables: $\sin(\theta)$ and $\cos(\theta)$, allowing the angle to be recovered post-prediction without boundary discontinuity.

# 3. The "Bin-and-Refine" Classification Architecture
To stabilize training, the continuous regression task can be split into a hybrid classification-regression head:
* **Classification Head:** Divides the 360° circle into 36 bins (10° each) and predicts which bin the tab falls into using Cross-Entropy Loss.
* **Regression Head:** Predicts the tiny, continuous scalar offset (-5° to +5°) from the center of that chosen bin.

# 4. Overcoming Data Starvation via Synthetic Generation
A 70-image dataset (yielding ~300 tube crops) is insufficient for deep metric learning. To bypass manual annotation bottlenecks, a 3D CAD model of a microcentrifuge tube should be imported into a rendering engine like Blender. A Python script can then automate the generation of 10,000+ training images—randomizing backgrounds, lighting, and tube rotations—while simultaneously exporting a perfect, mathematically exact CSV of the 360° ground truth orientations.

## 📂 Repository Structure
* `notebook.ipynb`: The complete end-to-end pipeline (Data formatting, Augmentation, Training, Evaluation, and OpenCV annotation).
* `best.pt`: The trained YOLOv8n-OBB model weights.
* `final_predictions.csv`: The extracted center coordinates, bounding boxes, and angles.
