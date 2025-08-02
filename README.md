# PolygonColorizer-UNet
A conditional image-to-image translation model that takes a **grayscale polygon image** and a **color name** as input, and generates the same polygon **filled with the specified color** using a custom-built UNet.  
Developed as part of the Ayna ML Internship Assignment.

---

## 📌 Problem Statement

Given:
- A grayscale image containing a polygon shape.
- A target color (e.g., "red", "blue", etc.).

Generate:
- A full-color image of the same polygon filled with the **specified color**.

---

## 🚀 Approach

We implemented a **UNet-based architecture** from scratch in **PyTorch**, with an additional **conditioning mechanism** to inject the color input.

### 🔧 Architecture Features:
- **Input**: Grayscale image (1x128x128) + one-hot encoded color vector (10-dim).
- **Color Conditioning**: Embedded and spatially expanded to concatenate as an extra input channel.
- **UNet**: 4-layer encoder-decoder with skip connections and `Conv2d` feature extraction.
- **Auxiliary Classifier**: Predicts the color label as a classification task to reinforce learning.
- **Loss Function**:


## 🧠 Dataset
data/
├── inputs/ # Grayscale polygon images
├── outputs/ # Colored output images
└── data.json # Mapping file with keys: input_polygon, colour, output_image


### Color Map Used:
python
{
  "red": 0, "blue": 1, "green": 2, "yellow": 3, "black": 4,
  "white": 5, "cyan": 6, "magenta": 7, "orange": 8, "purple": 9
}

📊 Experiment Tracking
All experiments were tracked using Weights & Biases (wandb).
📎 wandb Project:
🔗 https://wandb.ai/basuchandril402-jadavpur-university-press/ayna-unet-colorization/runs/c5wnlbz6?nw=nwuserbasuchandril402

Key Hyperparameters:
| Parameter     | Value                    |
| ------------- | ------------------------ |
| Epochs        | 100                      |
| Batch Size    | 8                        |
| Learning Rate | 1e-3                     |
| Loss          | L1 + 0.2 \* CrossEntropy |
| Model         | UNet + conditioning      |
| Optimizer     | Adam                     |

🧪 Results
✅ Model trained on 56 samples over 100 epochs
✅ High visual accuracy on shape and color reproduction
✅ Predicted color class matches input in most cases

📷 Sample Prediction:
<img width="580" height="225" alt="image" src="https://github.com/user-attachments/assets/e9bc85bd-dfa6-47ee-b512-888528f0acd8" />

🛠️ Setup Instructions (Google Colab / Local)
1. Clone repo or upload .ipynb to Colab
2. Install required packages: pip install torch torchvision wandb
3. Download dataset: First link your Google drive to your Google Colab (if not done yet), then
!gdown --id 1QXLgo3ZfQPorGwhYVmZUEWO_sU3i1pHM --output dataset.zip
!unzip -q dataset.zip -d data
4. Train model: Run all cells in the notebook.
5. Log into wandb: Put the following code cell before building your Unet
import wandb
wandb.login()

📁 Files Included
| File                   | Description                            |
| ---------------------- | -------------------------------------- |
| unet_colorizer.ipynb | Full training + inference notebook     |
| PolygonColorDataset  | Custom PyTorch Dataset class           |
| UNet                 | Model with conditional color embedding |
| README.md            | This file                              |
| samples/             | Sample I/O images for demo  |



