# The-Food-Hazard-Detection

This repository contains code and experiments for **food hazard classification** using **BERT** and **RoBERTa**. The project focuses on **imbalanced data** and explores techniques such as **Easy Data Augmentation (EDA)**, **random oversampling**, and **focal loss** to improve classification performance on minority classes.

## Project Overview

Food hazard detection presents unique challenges due to:
- **Severe class imbalance** in hazard categories  
- **Short, unstructured text** in recall reports  
- **Overlapping semantic information** across categories  

To address these, we implement a **transformer-based** approach, fine-tuning BERT and RoBERTa with:
1. **EDA** for lexical augmentation  
2. **Random oversampling** to balance minority classes  
3. **Focal loss** to emphasize hard-to-classify samples  

## Code and Notebooks

- **`5018_project_all.ipynb`**:  
  Main Jupyter notebook containing:
  - Data loading and preprocessing  
  - Implementation of EDA and oversampling  
  - Focal loss integration  
  - Model training (BERT/RoBERTa)  
  - Evaluation with accuracy, F1-macro, and F1-weighted  

## How to Use

1. **Clone or download** this repository.
2. **Open** `5018_project_all.ipynb` in Jupyter or another notebook environment.
3. **Run each cell** in sequence:
   - **Preprocessing**: Tokenization, stopword removal, lemmatization, and outlier filtering.
   - **EDA / Oversampling**: Augment or rebalance the training data.
   - **Training**: Fine-tune BERT/RoBERTa with or without focal loss.
   - **Evaluation**: Compare accuracy and F1 scores across methods.

## Model Configurations

We compare multiple training configurations:

- **Baseline**: Standard BERT fine-tuning.  
- **BERT + Oversampling**: Increases minority-class samples.  
- **BERT + EDA**: Applies data augmentation without class rebalance.  
- **BERT + Focal Loss**: Dynamically reweights loss for misclassified samples.  
- **BERT + EDA + Focal Loss**: Combines augmentation with focal loss.  
- **RoBERTa**: Similar setups, substituting BERT with RoBERTa.


---

**Enjoy exploring our food hazard detection project!**

