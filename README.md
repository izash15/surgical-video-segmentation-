# Surgical Video Segmentation

**Expert Branches: Module Diversity for Stronger Feature Learning in Laparoscopic Segmentation**

This AI capstone project was completed at **Saint Louis University (SLU) in 2026**. The project builds on the original **TriPath expert branch architecture** developed by **Lin Guo, Chiara Camerota, Mohammad Mahmoud, and Flavio Esposito** for laparoscopic surgical image segmentation.

The original TriPath model used three expert branches to extract different types of features from surgical images. In this project, we extend that model into a **QuadPath architecture** by adding a fourth expert branch based on a pretrained **MiT-B0 Vision Transformer**. The goal is to increase feature diversity by combining local convolution-based features with transformer-based global context.

The long-term vision of this work is to support surgical scene understanding for robotic-assisted surgery systems such as the **Da Vinci Surgical System**. While real-time organ labeling is a possible future application, this project focuses on the foundational step: improving organ segmentation accuracy on still laparoscopic surgical images.

---

## 1. Project Overview

Laparoscopic surgical image segmentation is challenging because organs can have similar colors, soft deformable shapes, tool occlusions, bleeding, lighting changes, and specular reflections. These issues make it difficult for a model to accurately identify organ boundaries and separate one anatomical structure from the background.

The original TriPath model addressed this by using three expert branches:

1. **CNN Branch**  
   Captures local tissue texture, colors, edges, and regional visual patterns.

2. **DCN Branch**  
   Uses deformable convolutions to better handle nonrigid and misaligned structures, which are common in soft organs.

3. **DSC Branch**  
   Uses dynamic snake-style convolutions to better follow curved boundaries and tube-like anatomical shapes.

This project adds a fourth branch:

4. **MiT-B0 Vision Transformer Branch**  
   Captures broader global context by helping the model understand relationships across the full surgical image, not only nearby local pixels.

Together, the four branches form a **QuadPath expert branch model**. Each branch extracts a different type of information from the same input image. These features are then combined through the bottleneck and UNet-style skip connections before the decoder produces the final segmentation mask.

---

## 2. Problem Definition and Scoping

### 2.1 AI Task

This project is a **supervised image segmentation task**.

The model receives a laparoscopic surgical image as input and predicts a segmentation mask as output.

### Input

The input is a still laparoscopic surgical image from the **DSAD surgical dataset**.

Each image shows a surgical scene containing anatomical structures, tools, lighting artifacts, and possible occlusions.

### Output

The output is a predicted segmentation mask.

In this project, the task is organized as **single-organ binary segmentation**. This means the model predicts whether each pixel belongs to a target organ or to the background.

For example, for a liver segmentation task:

- Pixel belongs to liver → organ class
- Pixel does not belong to liver → background

### Goal

The goal is to improve the quality of predicted organ masks by adding a pretrained Vision Transformer branch to the existing TriPath expert branch architecture.

The main research question is:

> Will adding a pretrained MiT-B0 Vision Transformer as a global context expert improve laparoscopic organ segmentation accuracy?

---

## 3. Type of Machine Learning Task

This project is:

- **Supervised learning**  
  The model is trained using images paired with ground-truth segmentation masks.

- **Image segmentation**  
  The model predicts a class label for each pixel.

- **Binary segmentation per organ**  
  Each training task focuses on separating one organ from the background.

This project is not classification, regression, generation, reinforcement learning, or unsupervised learning.

---

## 4. Model Architecture

### 4.1 Baseline: TriPath Expert Branch Model

The baseline model is based on a three-branch architecture. Each branch acts as an expert feature extractor.

The three original branches are:

| Branch | Expert Role |
|---|---|
| CNN Branch | Regional texture and local visual patterns |
| DCN Branch | Nonrigid and misaligned organ structures |
| DSC Branch | Curved boundaries and anatomical shapes |

The purpose of using separate branches is to allow each branch to specialize in a different feature type instead of forcing one model path to learn everything.

### 4.2 Proposed Model: QuadPath

The proposed model adds a fourth expert branch:

| Branch | Expert Role |
|---|---|
| ViT / MiT-B0 Branch | Global image context and long-range relationships |

The final architecture becomes:

```text
Input Image
   |
   |--> CNN Branch  --> texture features
   |
   |--> DCN Branch  --> deformable structure features
   |
   |--> DSC Branch  --> curved boundary features
   |
   |--> ViT Branch  --> global context features
   |
   v
Feature Fusion / Bottleneck
   |
Decoder
   |
1x1 Convolution
   |
Segmentation Mask Output
```

### 4.3 Why MiT-B0?

MiT-B0 was chosen because it is a lightweight hierarchical Vision Transformer backbone. It provides global context while remaining computationally more manageable than larger transformer models.

The MiT-B0 branch helps the model consider the larger surgical scene. This is useful because some organs may look similar up close, but their location and surrounding structures can help identify them correctly.

In simple terms:

- CNN looks at local details.
- DCN handles shape deformation.
- DSC follows curved structures.
- MiT-B0 looks at the bigger picture.

---

## 5. Encoder, Decoder, Bottleneck, and Skip Connections

### Encoder

The encoder extracts features from the input image.

In this project, each expert branch acts like an encoder path. Each branch studies the same image in a different way and produces feature maps.

### Bottleneck

The bottleneck is where the deepest features from all four expert branches are combined.

It acts like a shared summary of what all experts learned from the image.

For example:

```text
CNN summary + DCN summary + DSC summary + ViT summary
```

These combined features are passed into the decoder.

### Decoder

The decoder rebuilds the segmentation mask from the learned features.

The decoder takes the compressed information from the bottleneck and gradually upsamples it back toward the original image resolution.

In simple terms:

> The encoder understands the image, and the decoder makes the prediction mask.

### Skip Connections

Skip connections pass earlier feature maps from the encoder branches directly to the decoder.

This is important because deep encoder features contain strong semantic information, but they can lose fine details such as edges and boundaries. Skip connections help bring back these details.

In this project, skip connections allow the decoder to use information from all four expert branches at matching resolution levels.

This helps the model produce more accurate organ masks.

---

## 6. Dataset

This project uses the **DSAD surgical dataset**, which contains laparoscopic surgical images with annotated organ segmentation masks.

The dataset includes multiple abdominal organs and anatomical structures, such as:

- abdominal wall
- colon
- inferior mesenteric artery
- intestinal veins
- liver
- pancreas
- small intestine
- spleen
- stomach
- ureter
- vesicular glands

The dataset was organized for single-organ binary segmentation. Each model run focuses on predicting one target organ against the background.

### Dataset Notes

The raw dataset should not be modified directly. Preprocessed images and masks should be saved into a separate processed data directory.

Example structure:

```text
data/
  raw/
    DSAD/
      images/
      masks/
  processed/
    DSAD/
      images/
      masks/
```

If the dataset cannot be shared publicly, users should obtain it from the official dataset provider or the course/project supervisor.

---

## 7. Data Preprocessing

The preprocessing pipeline prepares image-mask pairs for training.

Typical preprocessing steps include:

1. Load laparoscopic images and their corresponding masks.
2. Match each image with the correct organ mask.
3. Resize images and masks to the model input size.
4. Normalize images for compatibility with the pretrained MiT-B0 branch.
5. Save processed files without overwriting the raw dataset.

For the MiT-B0 branch, image normalization follows ImageNet-style normalization because the backbone was pretrained on ImageNet.

Example normalization:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

---

## 8. Evaluation Metrics

The model is evaluated using four main segmentation metrics:

| Metric | Better Direction | Purpose |
|---|---|---|
| mIoU | Higher is better | Measures overlap between predicted mask and ground-truth mask |
| mDice | Higher is better | Measures segmentation overlap and similarity |
| MASD | Lower is better | Measures average boundary distance error |
| NSD | Higher is better | Measures boundary agreement within an acceptable tolerance |

### 8.1 mIoU

Mean Intersection over Union measures how much the predicted mask overlaps with the ground-truth mask.

Higher mIoU means better segmentation overlap.

### 8.2 mDice

Mean Dice score also measures overlap between the predicted and true masks.

Higher mDice means the predicted organ region matches the ground truth more closely.

### 8.3 MASD

Mean Average Surface Distance measures how far the predicted boundary is from the ground-truth boundary.

Lower MASD is better because it means the predicted boundary is closer to the real boundary.

### 8.4 NSD

Normalized Surface Dice measures how much of the predicted boundary is within an acceptable distance from the ground-truth boundary.

Higher NSD means better boundary agreement.

---

## 9. Results Summary

The QuadPath model was compared against the original TriPathEB-WF model.

Average metric comparison:

| Metric | TriPathEB-WF Original | QuadPath Average |
|---|---:|---:|
| mIoU | 0.454 | 0.569 |
| mDice | 0.541 | 0.664 |
| MASD | 20.439 | 12.655 |
| NSD | 0.432 | 0.577 |

The QuadPath model improved:

- mIoU
- mDice
- NSD

It also reduced:

- MASD

This suggests that the added MiT-B0 branch improved both region overlap and boundary quality.

---

## 10. Assumptions and Limitations

### Assumptions

- The DSAD masks are treated as ground truth.
- Each organ segmentation task is treated as a binary segmentation problem.
- Improvements in still-image segmentation may support future progress toward video-based surgical guidance.
- The pretrained MiT-B0 features are useful for laparoscopic images even though the model was pretrained on natural images.

### Limitations

- This project currently works with still images, not live surgical video.
- The model does not yet perform real-time inference during surgery.
- The dataset size and organ class balance may affect performance.
- Some organs are harder to segment due to small size, visual similarity, or occlusion.
- Adding a fourth branch increases model complexity and computational cost.
- The model was evaluated experimentally and is not clinically validated.

---

## 11. How to Run the Project

### 11.1 Environment Setup

Create and activate a conda environment:

```bash
conda create -n expertbranches python=3.10
conda activate expertbranches
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Depending on the environment, additional packages may be needed for PyTorch, torchvision, transformers, and MMCV.

### 11.2 Data Preparation

Place the raw dataset in:

```text
data/raw/DSAD/
```

Then run the preprocessing script:

```bash
python src/data/dsad_preprocess.py \
  --data-root data/raw/DSAD \
  --output-root data/processed/DSAD
```

### 11.3 Training

Example training command:

```bash
python train_EB.py \
  --data-root data/processed/DSAD \
  --images-subdir images \
  --masks-subdir masks \
  --img-size 512 512 \
  --num-classes 2 \
  --epochs 60 \
  --batch-size 16 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --val-split 0.1 \
  --save-dir experiments/quadpath_dsad \
  --run-name quadpath_mitb0
```

### 11.4 Training With Mixed Precision

```bash
python train_EB.py \
  --data-root data/processed/DSAD \
  --img-size 512 512 \
  --num-classes 2 \
  --epochs 60 \
  --batch-size 16 \
  --amp \
  --save-dir experiments/quadpath_dsad \
  --run-name quadpath_mitb0_amp
```

---

## 12. Important Files

### `src/models/tripath.py`

Contains the original TriPath model components, including the CNN, DCN, and DSC branch modules.

### `src/models/ViT.py`

Contains the MiT-B0 encoder implementation using the pretrained SegFormer backbone.

### `src/models/ViTStacked.py`

Contains the QuadPath model that combines CNN, DCN, DSC, and MiT-B0 branches.

### `train_EB.py`

Main training script for running experiments, saving checkpoints, and logging metrics.

### `src/data/dsad_preprocess.py`

Preprocessing script for preparing DSAD images and masks.

---

## 13. Citation

If referencing the original Expert Branches paper, use:

```text
L. Guo, C. Camerota, M. Mahmoud, and F. Esposito, 
"Expert Branches: Module Diversity for Stronger Feature Learning in Laparoscopic Segmentation," 
Proceedings of Machine Learning Research, MIDL 2026, 2026.
```

---

## 14. Authors

This project was completed as part of an AI capstone at Saint Louis University.

**Students:**

- Isabelle Davis
- Ramez Mosad

**Supervisors:**

- Lin Guo
- Flavio Esposito
