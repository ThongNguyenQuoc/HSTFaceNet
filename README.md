# HSTFaceNet

## 1. Overview
ElasticArcFace is a powerful face recognition model that combines the advantages of ArcFace and ElasticNet regularization. It uses an advanced neural network architecture to provide highly accurate facial recognition results. This model leverages the latest advances in deep learning to achieve high performance in both accuracy and efficiency.

## 2. Installation:
To install the required dependencies, run the following command:

```shell
!pip install tensorboard
!pip install easydict
!pip install mxnet
!pip install onnx
!pip install scikit-learn
```



## 3. Usage
ElasticArcFace can be used for both training and inference. After installation, you can train the model on your dataset or use a pre-trained model for face recognition tasks. Detailed steps are provided in the "Train Model" section below.

## 4. Dataset
ElasticArcFace is compatible with various face recognition datasets such as GLINT360K, MS1M, and VGGFace2. You can prepare your dataset in a suitable format (e.g., image folders or a CSV file with image paths and labels) and specify the dataset path in the configuration file (e.g., configs/glint360k_r100.py).

## 5. Train model
### 5.1. Train model at ElasticFace
```shell
!CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train_Elastic.py 
```

### 5.2. Train model at AdaFace + PartialFC
```shell
!CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train_Elastic_pretrained.py --resume 1
```
