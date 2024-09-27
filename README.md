# Facial Reconstruction from CCTV Footage

## Links

[![Dataset](https://img.shields.io/badge/Kaggle-Dataset-1DA1F2?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/selfishgene/youtube-faces-with-facial-keypoints)
[![Whimsical](https://img.shields.io/badge/Project-Flow-FB8C00?style=for-the-badge&logo=whimsical&logoColor=white)](https://whimsical.com/ethos-24-iit-guwahati-4EKrywTGGDVYnw8zFqMWsW)
[![Project Paper](https://img.shields.io/badge/Tech-Paper-1DA1F2?style=for-the-badge&logo=google&logoColor=white)](https://docs.google.com/document/d/1Y6O3dmGomgiGL6AV6-QGcjhODdklS2AOA7uunwgv7UE/edit?usp=sharing)


# Project Structure

```plaintext
Facial-reconstruction-from-CCTV-footage/
├── 🟫 assets/
├── 🟥 models/
├── 🟨 output/
├── 🟧 .gitignore           
├── 🟦 ethos24_autoencoder_cctv_facial_recon.ipynb   
├── 🟩 README.md   
├── 🟪 requirements.txt       # Python dependencies for the project
├── 🔴 venv/                  # Virtual environment directory (ignored by Git)
└── 🔴 dataset/               # Folder for dataset files (ignored by Git)
```


## Overview
This project reconstructs human faces from low-quality CCTV footage using advanced machine learning techniques.

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
If you are using Jupyter Notebook, you'll want to make sure your virtual environment is selected as the kernel:

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
