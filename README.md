# 🎤 Vocal Technique Recognition 🗣️

**Final Report - MIE1517 Introduction to Deep Learning (2025 Fall) - Team 20**

![Team20](./Team.png)

## 📋 Project Overview

This project focuses on **Vocal Technique Recognition** using deep learning and transfer learning. The goal is to automatically classify different vocal singing techniques from audio recordings, which has applications in music education, vocal training, and audio analysis.

We developed a deep learning pipeline that can recognize **six distinct vocal techniques**:
- **Breathy**: Produces a soft, airy quality by allowing excess air to escape
- **Falsetto**: Uses a higher register with lighter vocal cord vibration
- **Vibrato**: Creates a slight pitch oscillation for expressiveness
- **Glissando**: Involves smooth pitch transitions between notes
- **Pharyngeal**: Uses throat resonance for a specific timbre
- **Mixed Voice**: Combines chest and head voice registers

## 🎯 Key Features

- **Transfer Learning Approach**: Uses pre-trained MERT (Music Audio Representation Transformer) model for feature extraction
- **Lightweight Architecture**: 1D-CNN classifier selected for optimal balance between accuracy and efficiency
- **Comprehensive Evaluation**: Tested 6 different model configurations (MERT/Mel-spectrogram × 1D-CNN/CRNN/Transformer)
- **Strong Performance**: Achieved **88.43% test accuracy** on the GTSinger dataset
- **Reproducible**: Complete code, trained models, and results included

## 🏗️ Model Architecture

Our final model combines:
1. **Frozen MERT Backbone**: Pre-trained audio representation model that extracts 1024-dimensional frame-level embeddings
2. **1D-CNN Classifier**: Lightweight convolutional neural network for temporal pattern recognition
3. **Pipeline**: Audio preprocessing → MERT feature extraction → 1D-CNN classification → 6-class output

**Key Finding**: The simpler 1D-CNN architecture outperformed more complex CRNN and Transformer classifiers, demonstrating that MERT embeddings already encode sufficient temporal context.

## 📊 Results

- **Test Accuracy**: 88.43%
- **Validation Accuracy**: ~86%
- **Best Model**: MERT + 1D-CNN (selected for best generalization performance)

Training curves and confusion matrices are available in the repository. See the [Training curves](./Training%20curves.png) for comparison of all 6 model configurations.

## 📁 Repository Structure

```
MIE1517-2025-Fall---Final-Report-Team20-/
├── README.md                          # This file
├── Final_Report_Team20.ipynb         # Main Jupyter notebook with complete walkthrough
├── Final_Report_Team20.html          # HTML export of the notebook
├── requirement.txt                    # Python package dependencies
├── Team20.png                        # Team logo
├── Training curves.png               # Comparison of all model training curves
├── GTsinger_metrics.png              # Dataset statistics visualization
│
├── data/                             # Dataset directory
│   ├── GTsinger.zip                  # (Download separately - see below)
│   └── demo_data/                    # Demo audio files for testing
│       ├── Breathy/
│       ├── Falsetto/
│       ├── Glissando/
│       ├── Mixed_voice/
│       ├── Pharyngeal/
│       └── Vibrato/
│
├── train/                            # Training results
│   └── GT_results/
│       ├── mert_cnn/                 # Final model (MERT + 1D-CNN)
│       ├── mert_crnn/                # MERT + CRNN results
│       ├── mert_transformer/         # MERT + Transformer results
│       ├── mel_cnn/                  # Mel-spectrogram + 1D-CNN
│       ├── mel_crnn/                 # Mel-spectrogram + CRNN
│       └── mel_transformer/          # Mel-spectrogram + Transformer
│
└── test/                             # Test results
    └── mert_cnn_test_results/
        └── confusion_matrix_normalized.png
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.7+
- CUDA-capable GPU (recommended for training)
- ~10GB free disk space for dataset

### 2. Installation

Clone this repository:
```bash
git clone https://github.com/stevekslee/MIE1517-2025-Fall---Final-Report-Team20-.git
cd MIE1517-2025-Fall---Final-Report-Team20-
```

Install required packages:
```bash
pip install -r requirement.txt
```

**Note**: For PyTorch installation, please find a compatible version for your environment:
- Visit [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- Select your CUDA version and install accordingly

### 3. Download Dataset

The project uses the **GTSinger** dataset [Zhang et al., 2024](https://arxiv.org/pdf/2409.13832). We processed the dataset to include only English singing `.wav` files.

**Download the processed dataset:**
1. Download from [GTsinger Dataset - Team20](https://drive.google.com/file/d/11aQMAexLnb_Qdb232ytvW-W8ImTae6BC/view?usp=sharing)
2. Place `GTsinger.zip` in the `data/` folder:
   ```
   data/
   └── GTsinger.zip
   ```
3. The notebook will automatically extract and process the dataset

### 4. Run the Notebook

Open and run the Jupyter notebook:
```bash
jupyter notebook Final_Report_Team20.ipynb
```

The notebook is organized as a step-by-step walkthrough:
- **Section 1**: Project description and approach
- **Section 2**: Implementation details
- **Section 3**: Model training and evaluation
- **Section 4**: Testing on new data
- **Section 5**: Related work and references
- **Section 6**: Discussion and insights

## 📖 Usage

### Training a Model

The notebook includes code to train all 6 model configurations. To train the final MERT + 1D-CNN model:

1. Follow the data loading steps in Section 3-1 and 3-2
2. Run the training code in Section 3-8 (Model 4: MERT + 1D-CNN)
3. Training results will be saved in `train/GT_results/mert_cnn/`

### Testing on New Data

To test the trained model on new audio files:

1. Load a trained model checkpoint (Section 4-1)
2. Use the test function (Section 4-2)
3. Test on the test dataset (Section 4-3) or demo files (Section 4-5)

### Using Pre-trained Models

Pre-trained model checkpoints are available in `train/GT_results/mert_cnn/best_model.pth`. Load and use them as shown in Section 4-1 of the notebook.

## 📈 Model Comparison

We evaluated 6 different model configurations:

| Feature Extractor | Classifier | Validation Accuracy | Notes |
|------------------|------------|---------------------|-------|
| MERT (frozen) | 1D-CNN | **~86%** | ✅ **Selected as final model** |
| MERT (frozen) | CRNN | ~84% | Overfitting observed |
| MERT (frozen) | Transformer | ~83% | Overfitting observed |
| Mel-spectrogram | 1D-CNN | ~75% | Baseline comparison |
| Mel-spectrogram | CRNN | ~73% | Baseline comparison |
| Mel-spectrogram | Transformer | ~72% | Baseline comparison |

**Key Insight**: Transfer learning with MERT significantly outperformed mel-spectrogram features, and the simpler 1D-CNN classifier achieved the best balance of accuracy and efficiency.

## 🔬 Technical Details

- **Audio Preprocessing**: Mono conversion, resampling to 24 kHz, normalization, fixed-length padding/truncation (10 seconds)
- **Feature Extraction**: MERT-v1-95M model (frozen) outputs 1024-dimensional frame-level embeddings
- **Classification**: 1D-CNN with global average pooling and fully connected layers
- **Training**: Cross-entropy loss, Adam optimizer, train/val/test split (70%/15%/15%)
- **Evaluation**: Accuracy, confusion matrix, per-class precision/recall/F1-score

## 📚 Dataset

- **Source**: [GTSinger Dataset](https://arxiv.org/pdf/2409.13832) by Zhang et al., 2024
- **Subset Used**: English singing samples only
- **Classes**: 6 vocal techniques (Breathy, Falsetto, Vibrato, Glissando, Pharyngeal, Mixed Voice)
- **Split**: 70% train / 15% validation / 15% test (stratified by singer to prevent data leakage)

## 📝 Report

The complete final report is available as:
- **Jupyter Notebook**: `Final_Report_Team20.ipynb` (interactive, runnable)
- **HTML Export**: `Final_Report_Team20.html` (static, viewable in browser)

The report includes:
- Complete project walkthrough from data loading to prediction
- Model architecture diagrams and explanations
- Training results and learning curves
- Test results and confusion matrices
- Example predictions on new data
- Discussion of insights and findings
- Related work and references

## 🎓 Course Information

- **Course**: MIE1517 - Introduction to Deep Learning
- **Term**: 2025 Fall
- **Institution**: University of Toronto
- **Team**: Team 20

## 📄 License

This project is submitted as part of academic coursework. Please refer to the original dataset licenses (GTSinger) for dataset usage terms.

## 🙏 Acknowledgments

- **GTSinger Dataset**: [Zhang et al., 2024](https://arxiv.org/pdf/2409.13832)
- **MERT Model**: [Music Audio Representation Transformer](https://github.com/yizhilll/MERT)
- **Course Instructors**: MIE1517 Teaching Team

## 📧 Contact

For questions or issues, please open an issue on this repository.

---

**Note**: This repository contains the complete code, trained models, and results for reproducibility. The notebook is designed to be a step-by-step guide that a fellow classmate can follow to understand and reproduce the key aspects of the project.
