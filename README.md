# OCR Handwritten Text

Welcome to the repository for my Engineering Thesis project! Here, I'm working on developing an Optical Character Recognition (OCR) system aimed at accurately recognizing handwritten text from various sources, with the ultimate goal of creating a unified system capable of handling diverse text data. Currently, the architecture is based on the CRNN model with CTC Loss.

## Project Overview

The primary objective of this project is to create an OCR system capable of accurately recognizing handwritten text. Here's a brief overview of the project's current status and future goals:

- **Current Status**: The OCR system has achieved the following results: 100% accuracy on two publicly available CAPTCHA datasets and a 75% accuracy rate on the IAM dataset (word segmented).

## To-Do List (OCR Part)

- **Replace Greedy Argmax Decoder**: Implement beam search decoder for improved accuracy.
- **Explore Pretrained Encoder Models**: Test and integrate all possible pretrained encoder models, including Visual Transformer.
- **Utilize CER**: Implement Character Error Rate for more comprehensive evaluation.
- **Fine-Tune on Polish Characters**: Generate synthetic data and fine-tune the system to recognize Polish characters effectively.

## To-Do List (End-to-End System)

- **Word and Line Segmentation**: Develop algorithms for word and line segmentation.
- **Data Preprocessing**: Implement data preprocessing techniques to improve OCR accuracy.
- **Dockerize**: Containerize the entire system using Docker (because I want to know how to use it).
