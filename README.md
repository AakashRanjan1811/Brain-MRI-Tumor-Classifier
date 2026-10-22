# Brain MRI Tumor Classifier

A deep learning project using **Convolutional Neural Networks (CNNs)** to classify brain MRI images into four categories: Glioma, Meningioma, Pituitary Tumor, and No Tumor.

## Tech Stack
- **Language:** Python
- **Frameworks:** TensorFlow, Keras
- **Libraries:** NumPy, scikit-learn, Matplotlib

## Dataset
- **Source:** [Brain Tumor MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Training Images:** 5,600
- **Testing Images:** 1,600
- **Total:** 7,200 MRI images

## Model Architecture
Built a CNN using the Keras Sequential API with:
- Conv2D and ReLU activation for feature extraction
- MaxPooling2D for spatial downsampling
- Dropout layers to reduce overfitting
- Dense layers with Softmax for four-class classification

**Training:** Adam optimizer, Categorical Crossentropy loss, 20 epochs.

## Model Performance

| Metric | Result |
|---|---|
| Training Accuracy | 96.66% |
| Test Accuracy | **85.13%** |
| Macro Precision | 85.82% |
| Macro Recall | 85.13% |
| Macro F1-Score | 84.89% |

## Future Improvements
- Data augmentation and hyperparameter tuning
- Transfer learning with pretrained CNN architectures
- Improved regularization to reduce overfitting

## Acknowledgments
Dataset: [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset). Credit any external notebook or code used as a starting point.
