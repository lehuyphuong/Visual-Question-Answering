# Visual-Question-Answering

This project implements a Visual Question Answering (VQA) system that takes an image and a natural language question as input and predicts the most plausible answer. Two approaches are explored:
- A hybrid CNN + LSTM architecture.
- A transformer-based model using ViT + RoBERTa.

*⚠️ Note: This project is designed as a template/demo implementation and has not been executed yet. The primary aim is to provide a step-by-step structure for building and training a VQA model.*

## 📄 Disclaimer
This project is based on coursework and materials from AIO2024. The implementation and documentation presented here are a personal learning exercise, created by independently analyzing, reorganizing, and summarizing ideas for educational purposes only. No commercial use or redistribution is intended.

## 📁 Project Structure
├── CNN_LSTM.ipynb         # ResNet + LSTM

├── ViT_RoBERTa.ipynb       # ViT + RoBERTa

## 📌 Key Features
1. CNN + LSTM (in CNN_LSTM.py)
    - Image encoder: ResNet18 from timm.
    - Text encoder: Token embedding + 2-layer bidirectional LSTM.
    - Fusion: Concatenate image and question embeddings.
    - Classifier: Two-layer MLP with GELU and dropout.
    - Training loop: Custom fit() and evaluate() functions with learning rate scheduler.

2. ViT + RoBERTa (in ViT_RoBERTa.py)
    - Image encoder: ViTModel from HuggingFace.
    - Text encoder: RoBERTa base model.
    - Fusion: Concatenate pooled embeddings.
    - Classifier: MLP on concatenated features.
    - Tokenizer/Processor: HuggingFace AutoTokenizer and ViTImageProcessor.

## 🧠 Dataset Preparation

The dataset is expected in the format of .txt files (vaq2.0.TrainImages.txt, etc.), where each line follows:
```
image_path \t question? answer
```
Example: 
```
COCO_val2014_000000391895.jpg\tWhat is the boy holding? A kite
```
## 🏗️ Model Overview
1. CNN + LSTM Architecture
    ```
    img_features = ResNet18(img)
    question_features = BiLSTM(Embedding(tokens))
    combined = concat(img_features, question_features)
    output = MLP(combined)
    ```

2. ViT + RoBERTa Architecture
    ```
    img_features = ViTProcessor(image)
    question_features = RoBERTaTokenizer(question)
    output = MLP(concat(img_features, question_features))
    ```

## 📝 Citation & Reference
- ResNet from timm
- ViT & RoBERTa from HuggingFace Transformers