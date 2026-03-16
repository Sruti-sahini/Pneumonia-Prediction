# Pneumonia Prediction Using Chest X-Ray Images

## Introduction

Pneumonia is a potentially life-threatening lung infection, especially dangerous for young children and the elderly. Early and accurate diagnosis is crucial for effective treatment. This project leverages deep learning to automatically detect pneumonia from chest X-ray images, aiming to support radiologists and improve diagnostic speed and accuracy.

We implement and compare three deep learning architectures:
1. **Multilayer Perceptron (MLP)** – Baseline model
2. **Convolutional Neural Network (CNN)**
3. **Residual Neural Network (ResNet, via transfer learning)**

We also apply Explainable AI (XAI) techniques to interpret model predictions.


## Dataset

- **Source:** [Chest X-Ray Images (Pneumonia) on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Total Images:** 5,863 pediatric chest X-rays (JPEG)
- **Categories:** Normal and Pneumonia
- **Splits:** Train, Test, and Validation sets

**Note:** The original validation set is very small, so a custom validation set is created from the training data.


## Setup & Requirements

**Python Version:** 3.8+  
**Key Libraries:**
- TensorFlow 2.x
- Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- scikit-learn
- Pillow, OpenCV

**To install requirements:**
```bash
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn pillow opencv-python tqdm
```

## Data Exploration & Preprocessing
**Class Distribution:**
- Normal: 1,583 images (27%)
- Pneumonia: 4,273 images (73%)
- The dataset is imbalanced.

**Image Size Analysis:**
- Normal: Avg. 1630×1346 px
- Pneumonia: Avg. 1204×826 px
- Images are resized to 224×224 px for model input.

**Preprocessing Steps:**
- Resize images to 224×224 px
- Normalize pixel values to [0, 1]
- Data augmentation (rotation, shift, shear, zoom, flip) for training
- Custom validation set (20% of training data)
- Class weights to address imbalance

## Model Architectures

### 1. Multilayer Perceptron (MLP)
- Baseline model with fully connected layers  
- Input: Flattened image (150,528 features)  
- 3 hidden layers with batch normalization and dropout  

### 2. Convolutional Neural Network (CNN)
- 4 convolutional blocks with max pooling and batch normalization  
- Dense layers with dropout for regularization  

### 3. Residual Neural Network (ResNet)
- Transfer learning using pre-trained ResNet50 (ImageNet)  
- Custom classification head  
- Fine-tuning of top layers after initial training  



## Training & Evaluation

- **Optimizer:** Adam (learning rate 0.0001)  
- **Loss:** Binary cross-entropy  
- **Callbacks:** Early stopping, learning rate reduction, model checkpointing  
- **Metrics:** Accuracy, Precision, Recall, F1-score, ROC AUC  

### Training Times:
- **MLP:** ~1,100 seconds  
- **CNN:** ~3,534 seconds  
- **ResNet:** ~1,138 seconds  

### Evaluation:
Models are evaluated on the test set using:
- Confusion matrix  
- ROC curve  
- Precision-recall curve  



## Explainable AI (XAI)

### Occlusion Sensitivity
Visualizes which regions of the X-ray most influence the model’s prediction by systematically occluding parts of the image and measuring prediction changes.

### Heatmaps
Generated for both CNN and ResNet to compare focus areas and interpret model decisions.



## Results

| Model | Accuracy | ROC AUC | F1 (Pneumonia) | Precision (Pneumonia) | Recall (Pneumonia) | Training Time (s) |
|-------|----------|---------|----------------|------------------------|--------------------|-------------------|
| MLP   | 76.3%    | 0.832   | 0.805          | 0.827                  | 0.785              | 1,101             |
| CNN   | 87.2%    | 0.936   | 0.896          | 0.906                  | 0.887              | 3,534             |
| ResNet| 86.7%    | 0.936   | 0.897          | 0.872                  | 0.923              | 1,138             |

- **CNN** achieved the highest accuracy and balanced performance.  
- **ResNet** performed slightly better on pneumonia cases (recall).  
- **MLP** baseline was significantly outperformed by both CNN and ResNet.  

### Visualizations:
- Training curves (accuracy, loss, precision, recall)  
- Confusion matrices  
- ROC and precision-recall curves  
- Model comparison bar charts  
- Occlusion sensitivity heatmaps  



## Key Findings

- Deep learning models (CNN, ResNet) can effectively detect pneumonia from chest X-rays.  
- CNN provided the best balance of accuracy, precision, and recall.  
- ResNet (transfer learning) showed strong generalization and slightly better recall for pneumonia.  
- Explainable AI (occlusion sensitivity) revealed that both models focus on clinically relevant lung regions.  
- The models could potentially assist radiologists in screening and diagnosis.  


## Conclusion & Future Work

### Conclusion:
Deep learning, especially CNNs and transfer learning with ResNet, are powerful tools for medical image classification. Explainable AI techniques are essential for clinical trust and adoption.

### Future Work:
- Test on larger and more diverse datasets  
- Implement additional XAI methods (e.g., Grad-CAM)  
- Develop a deployment pipeline for clinical use  
- Extend to multi-class classification (e.g., viral vs. bacterial pneumonia)  


## References

- [Chest X-Ray Images (Pneumonia) dataset (Kaggle)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)  
- Rajpurkar, P., et al. (2017). *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning.*  
- Zeiler, M. D., & Fergus, R. (2014). *Visualizing and Understanding Convolutional Networks.*  
- He, K., et al. (2016). *Deep Residual Learning for Image Recognition.*





