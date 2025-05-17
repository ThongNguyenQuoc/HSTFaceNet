# HSTFaceNet

## 📋 Table of Contents

1. [Overview](#1-overview)  
2. [Installation](#2-installation)  
3. [Attention Fix for MXNet (Optional)](#3-attention-fix-for-mxnet-optional)  
4. [Dataset Preparation](#4-dataset-preparation)  
5. [Training](#5-training)  
    - [5.1 Train ElasticFace](#51-train-elasticface-baseline)  
    - [5.2 Train ElasticFace + AdaFace + PartialFC](#52-train-elasticface--adaface--partialfc)  
    - [5.3 Train ElasticFace + AdaFace + Self-Attention](#53-train-elasticface--adaface--self-attention-enhanced-model)  
6. [Evaluation](#6-evaluation)  
7. [Results](#7-results)  
8. [References](#8-references)  

---

## 1. Overview

ElasticArcFace is an advanced face recognition framework that integrates:

- **ArcFace (Additive Angular Margin Loss)**
- **ElasticNet Regularization (ElasticFace)**
- **AdaFace (Quality Adaptive Margin)**
- **Self-Attention modules** (for enhanced feature extraction)

The goal is to improve recognition accuracy, especially in difficult conditions such as pose variations, occlusions, or low-quality images.

---

## 2. Installation

Install the required dependencies:

```bash
!pip install tensorboard
!pip install easydict
!pip install mxnet
!pip install onnx
!pip install scikit-learn
```

## 3. Attention Fix for MXNet (Optional)
For some versions of MXNet, you might encounter compatibility warnings regarding bool.
Fix it using the following command:

```bash
sed -i 's/bool = onp.bool/bool = bool/' /usr/local/lib/python3.11/dist-packages/mxnet/numpy/utils.py

```

## 4. Dataset Preparation

ElasticArcFace is compatible with various face recognition datasets such as:

- **MS1MV2**

You can use the public datasets provided by [InsightFace Datasets Repository](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_).

### Example Datasets Link:
- 📦 MS1MV2  (List & Download Scripts):  
  https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_/

### Dataset Structure Example:

