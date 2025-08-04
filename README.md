# Machine Learning
# CNN-Based Classification of Eye Diseases

This project builds a Convolutional Neural Network (CNN) to classify **retinal fundus images** into three categories: **Normal**, **Cataract**, and **Glaucoma**. These eye conditions are among the leading causes of visual impairment globally, and early detection is critical for effective treatment and prevention of vision loss.

Manual screening of fundus images is time-consuming, subjective, and often limited by access to trained ophthalmologists. This project explores the use of deep learning — specifically CNNs — to automate the detection and classification process with high accuracy, offering a scalable and objective alternative in clinical and remote settings.

---

## Overview
The CNN is trained on a pre-labeled dataset of fundus images. The pipeline includes:
- Image preprocessing.
- Model architecture definition and training. 
- Evaluation using classification metrics and visualizations.  
- Interpretation of results and performance reporting.

---

## Objectives
- Build a CNN model to classify retinal images into Normal, Cataract, or Glaucoma.  
- Train and validate the model using real patient data.  
- Evaluate performance with accuracy, precision, recall, F1-score, and confusion matrix. 

---

## Tools & Technologies
- **Programming**: Python (Keras, TensorFlow, scikit-learn, OpenCV)  
- **Model**: Custom CNN architecture (Conv2D, MaxPooling, Flatten, Dense layers)  
- **Metrics**: Confusion matrix, classification report, accuracy plot  
- **Environment**: Jupyter Notebook / Python script executed locally

## Notes
- Retinal images were provided by King's university as part of the MSc Applied Bioinformatics coursework project.
- This project is for educational and research purposes only and is not intended for clinical deployment without further validation.
