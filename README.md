# Facial Reconstruction from CCTV Footage


## Links

[![Dataset](https://img.shields.io/badge/Kaggle-Dataset-1DA1F2?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/selfishgene/youtube-faces-with-facial-keypoints)
[![Whimsical](https://img.shields.io/badge/Project%20Flow-Flow-FB8C00?style=for-the-badge&logo=whimsical&logoColor=white)](https://whimsical.com/ethos-24-iit-guwahati-4EKrywTGGDVYnw8zFqMWsW)
[![Project Paper](https://img.shields.io/badge/Tech%20Paper-Link-1DA1F2?style=for-the-badge&logo=google&logoColor=white)](https://docs.google.com/document/d/1Y6O3dmGomgiGL6AV6-QGcjhODdklS2AOA7uunwgv7UE/edit?usp=sharing)


## ⚠️ Warnings

### 🚧 Proof of Concept (POC) in Development
This project is a **Proof of Concept (POC)** and is still in active development. It is **not a fully functional model** and should **not be used for any professional or real-world applications** at this stage. Please use this project solely for **development and educational purposes**.

### ⚠️ Do Not Use for Critical Tasks
The model is currently under development and its predictions **should not be relied upon for sensitive tasks** such as legal or security-related applications. Future improvements and refinements are planned, but the current state is experimental.

---

# Project Structure

```plaintext
Facial-reconstruction-from-CCTV-footage/
├── 🟫 assets/                 # Assets like images, diagrams
├── 🟥 models/                 # Saved models during training
├── 🟨 output/                 # Outputs such as predictions and comparisons
├── 🟧 .gitignore              
├── 🟦 ethos24_autoencoder_cctv_facial_recon.ipynb  # Main notebook
├── ⬜ LICENSE  
├── 🟩 README.md   
├── 🟪 requirements.txt        # Python dependencies for the project
├── 🔴 venv/                   # Virtual environment directory (ignored by Git)
└── 🔴 dataset/                # Folder for dataset files (ignored by Git)
```

## Overview

This project aims to reconstruct human faces from low-quality CCTV footage using deep learning techniques, focusing on overcoming challenges such as low resolution, motion blur, noise, and poor lighting conditions.

---

## 📌 Important Note: Made for Ethos 2024 Hackathon

This project was created as part of the **Saptang Labs - Machine Learning Challenge: Ethos 2024** hackathon conducted by the **Indian Institute of Technology (IIT), Guwahati**.

### Problem Statement

The challenge focuses on building a **facial reconstruction model** capable of enhancing low-quality CCTV footage to assist in **identifying suspects** in various scenarios. The key difficulties include handling **motion blur**, **low resolution**, and **poor lighting**, all of which significantly degrade the visual quality of the footage. This project aims to address these issues through techniques such as **noise reduction**, **super-resolution**, and **image enhancement**.

---

## Table of Contents

- [Project Setup Guide](#project-setup-guide)
- [Model Architecture and Explanation](#model-architecture-and-explanation)
- [Preprocessing Steps](#preprocessing-steps)
- [Model Evaluation and Metrics](#model-evaluation-and-metrics)
- [Prediction Results](#prediction-results)
- [Bias distribution and Visualization](#model-bias-and-distribution)
- [Future Work and Enhancements](#future-works-and-enhancements)
- [Acknowledgments](#acknowledgments)

---

## Project Setup Guide

### Step 1: Clone the Repository

To get started, clone the repository from GitHub. Open your terminal and run the following command:

```bash
git clone https://github.com/shivangichaudhary/Facial-reconstruction-from-CCTV-footage.git
cd Facial-reconstruction-from-CCTV-footage
```

### Step 2: Create a Virtual Environment

Set up a Python virtual environment to manage project dependencies:

```bash
python -m venv venv
```
OR
```bash
python3 -m venv venv
```

### Step 3: Activate the Virtual Environment

- **Windows**:
```bash
.\venv\Scripts\activate
```

- **macOS/Linux**:
```bash
source venv/bin/activate
```

### Step 4: Install Dependencies

Install all the necessary Python libraries and dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Step 5: Download the Dataset

The dataset is not included in the repo. To download the dataset:
1. Go to the [YouTube Faces with Facial Keypoints](https://www.kaggle.com/datasets/selfishgene/youtube-faces-with-facial-keypoints) dataset page.
2. Download the dataset, extract its contents, and place them in the `dataset/` folder.
3. Ensure that the dataset folder is structured correctly for the project to access.

### Step 6: Set the Virtual Environment for Jupyter Notebook (Optional)

If you are using Jupyter Notebook, make sure your virtual environment is selected as the kernel:

1. Install ipykernel:
```bash
pip install ipykernel
```

2. Add the virtual environment to Jupyter:
```bash
python -m ipykernel install --user --name=venv
```

3. In Jupyter Notebook, select the kernel named `venv`.

### Step 7: Run the Project

Now you are ready to run the project. Open the Jupyter Notebook:

```bash
jupyter notebook ethos24_autoencoder_cctv_facial_recon.ipynb
```
Follow the steps in the notebook to execute the project.

---

## Model Architecture and Explanation

The project uses a **Convolutional Autoencoder** based on a modified **LeNet-style architecture**. The architecture consists of:

- **Encoder**:
  - Series of convolution layers with max-pooling to reduce the spatial dimensions while capturing features.
- **Latent Space**:
  - The compressed representation that encodes the essential features for face reconstruction.
- **Decoder**:
  - Mirrors the encoder to upsample and reconstruct the input image.
- **Skip Connections**:
  - Added from the encoder to the decoder to preserve fine details during reconstruction.
- **Activation Functions**:
  - `ReLU` for convolution layers and `sigmoid` for the output layer to produce pixel values between 0 and 1.

### Architecture
![Autoencoder Architecture](https://raw.githubusercontent.com/shivangichaudhary/Facial-reconstruction-from-CCTV-footage/refs/heads/main/assets/autoencoder_architecture.png)

---

## Preprocessing Steps

### **Low-Light Enhancement**

The CCTV footage often suffers from low lighting. To mitigate this:

- **Gamma Correction**: Enhances low-light regions by applying `exposure.adjust_gamma(img, gamma=0.8)` which brightens darker areas while maintaining image contrast.

### **Noise Reduction**

Noise is inherent in low-quality footage. A denoising method is used:

- **Non-local Means Denoising**: Applied using `cv2.fastNlMeansDenoisingColored` to reduce chromatic and luminance noise in the image while preserving detail.

### **Super-Resolution**

While not explicitly super-resolution, the **upsampling layers** in the decoder help increase image size after encoding, enhancing spatial detail:

- **Upsampling2D Layers**: Increases the resolution of the encoded feature maps to reconstruct a higher-quality image.

### **Data Augmentation**

To generalize the model and reduce overfitting, augmentation techniques are applied:

- **Horizontal Flipping**: Randomly flips images horizontally with a 50% probability to simulate different orientations of faces.

---

## Model Evaluation and Metrics

The model's performance is evaluated using the following metrics:

1. **Mean Squared Error (MSE)**: Measures the difference between the original and reconstructed images.
2. **Peak Signal-to-Noise Ratio (PSNR)**: Calculates the ratio between the maximum possible power of a signal and the power of corrupting noise.

### Performance metrics

|  | Metric                         | Value     |
|--|---------------------------------|-----------|
| 0 | Mean Squared Error (MSE)        | 0.000480  |
| 1 | Peak Signal-to-Noise Ratio (PSNR) | 33.188666 |


---

## Prediction Results

After training, the reconstructed images are compared against the original images. Visual comparisons can be found in the `output/` folder.
|  | Filename        | Original   | Reconstructed   |
|--|-----------------|------------|-----------------|
| 0 | Abdullah Gul    | `[[0.3333333333333333, 0.19607843137254902, 0....` | `[[0.3485127, 0.3368767, 0.2930863], [0.318771...` |
| 1 | George Harrison | `[[0.4627450980392157, 0.3058823529411765, 0.2...` | `[[0.4017781, 0.39366928, 0.35159398], [0.3649...` |
| 2 | Fidel Castro    | `[[0.3333333333333333, 0.19607843137254902, 0....` | `[[0.34945077, 0.3377307, 0.2939164], [0.32003...` |
| 3 | Talisa Soto     | `[[0.28627450980392155, 0.2627450980392157, 0....` | `[[0.3513748, 0.37173614, 0.33502895], [0.3133...` |
| 4 | Andrew Bernard  | `[[0.2901960784313726, 0.2627450980392157, 0.2...` | `[[0.35628965, 0.37772307, 0.34051785], [0.320...` |

### Comparison Images

<img src="https://raw.githubusercontent.com/shivangichaudhary/Facial-reconstruction-from-CCTV-footage/refs/heads/main/output/Abdullah%20Gul_comparison_0.png" alt="Abdullah Gul Comparison" width="512" height="256">
<img src="https://raw.githubusercontent.com/shivangichaudhary/Facial-reconstruction-from-CCTV-footage/refs/heads/main/output/George%20Harrison_comparison_3.png" alt="George Harrison Comparison" width="512" height="256">

*(Note: The model is in development phase. so the results may not good. change the param of the model for more accurate prediction or wait for our official release)*

### Model Bias and distribution

| Layer Name    | Bias Values                                           |
|---------------|------------------------------------------------------|
| `conv2d`     | [ 0.0597,  0.1302,  0.0454,  0.0412,  0.0985,  0.0651,  0.0463,  0.0315,  0.0454, -0.0308,  0.0516, -0.0126,  0.0475,  0.0242,  0.0539, -0.0247,  0.0575, -0.0478,  0.0324,  0.0006,  0.0553,  0.0124, -0.0346,  0.0524,  0.0466,  0.0030,  0.0362,  0.0580, -0.0038,  0.0374, -0.0373, -0.0488] |
| `conv2d_1`   | [ 0.0490, -0.0137,  0.0626, -0.0467,  0.0350,  0.0610,  0.0516,  0.0377,  0.0170,  0.0522,  0.0479, -0.0188, -0.0423,  0.0604, -0.0083, -0.0463] |
| `conv2d_2`   | [-0.0414, -0.0272,  0.0376, -0.0130, -0.0016, -0.0488,  0.0510, -0.0060,  0.0623,  0.0594, -0.0164,  0.0682,  0.0121,  0.1069, -0.0485,  0.0485] |
| `conv2d_3`   | [ 0.0521,  0.0476,  0.0563, -0.0477, -0.0612,  0.0027, -0.0374,  0.0523,  0.0591,  0.0199,  0.0055,  0.0547,  0.0556, -0.0334, -0.0146, -0.0326, -0.0384, -0.0214,  0.0532, -0.0165, -0.0127,  0.0625, -0.0529,  0.0671, -0.0428, -0.0220,  0.0604,  0.0627, -0.0413,  0.1004, -0.0652, -0.0463] |
| `conv2d_4`   | [-0.0484, -0.0425, -0.0461]                         |

### Visualizations

![Bias value distribution Hist](https://raw.githubusercontent.com/shivangichaudhary/Facial-reconstruction-from-CCTV-footage/refs/heads/main/assets/autoencoder.bias_dist.png)


---

## Future Works and Enhancements
- **Live CCTV Video Processing:** Extend the model to handle real-time CCTV video feeds for on-the-fly facial reconstruction.
- **Incorporate 3D Landmarks:** Utilize 3D facial landmarks to improve the accuracy and realism of facial reconstruction.
- **Multi-Scale Feature Learning:** Implement multi-scale convolutions to capture fine details at various scales, enhancing the quality of the reconstructed faces.
- **Advanced Noise Reduction:** Develop more sophisticated noise reduction techniques to handle noisy input images from low-quality CCTV footage.
- **Improved Super-Resolution:** Integrate state-of-the-art super-resolution models, such as SRCNN or ESRGAN, to further enhance the quality of reconstructions from extremely low-resolution images.
- **Robustness to Motion Blur:** Improve the model’s ability to process severely blurred images by developing specific techniques to handle motion blur.
- **Experiment with Super-Resolution Networks:** Continue exploring advanced super-resolution networks for better face reconstruction results, particularly with extremely low-quality images.

---

## Acknowledgments

Special thanks to the dataset contributors and open-source tool developers.

---
