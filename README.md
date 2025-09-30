# **CIFAR10 Classification Using Convolutional Neural Network**

## **Objective**
The main objective of this assignment is to create a model consisting of multiple Convolution layers that can accurately classify unseen images from the CIFAR-10 dataset into their respective classes. The network should follow:
- works on CIFAR-10 Dataset
- has the architecture to C1C2C3C40 (No MaxPooling, but convolutions, where the last one has a stride of 2 instead) (NO restriction on using 1x1) (If you can figure out how to use Dilated kernels here instead of MP or strided convolution, then 200pts extra!)
- total RF must be more than 44
- One of the layers must use Depthwise Separable Convolution
- One of the layers must use Dilated Convolution
- use GAP (compulsory):- add FC after GAP to target #of classes (optional)
- Use the albumentation library and apply:
  - horizontal flip
  - shiftScaleRotate
  - coarseDropout (max_holes = 1, max_height=16px, max_width=16, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
- achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k.

## **Dataset Overview**
The CIFAR-10 dataset contains 60,000 32×32 colored images across 10 classes (6,000 images per class):
- Training set: 50,000 images
- Test set: 10,000 images

## **Data Exploration**

## **Experiments**

## **Final Architecture**


### &emsp; **Model Summary**

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 32, 32]             864
       BatchNorm2d-2           [-1, 32, 32, 32]              64
              ReLU-3           [-1, 32, 32, 32]               0
           Dropout-4           [-1, 32, 32, 32]               0
            Conv2d-5           [-1, 64, 16, 16]          18,432
       BatchNorm2d-6           [-1, 64, 16, 16]             128
              ReLU-7           [-1, 64, 16, 16]               0
           Dropout-8           [-1, 64, 16, 16]               0
            Conv2d-9           [-1, 16, 16, 16]           1,024
      BatchNorm2d-10           [-1, 16, 16, 16]              32
             ReLU-11           [-1, 16, 16, 16]               0
          Dropout-12           [-1, 16, 16, 16]               0
           Conv2d-13           [-1, 32, 16, 16]           4,608
      BatchNorm2d-14           [-1, 32, 16, 16]              64
             ReLU-15           [-1, 32, 16, 16]               0
          Dropout-16           [-1, 32, 16, 16]               0
           Conv2d-17             [-1, 32, 8, 8]             288
           Conv2d-18             [-1, 16, 8, 8]             512
      BatchNorm2d-19             [-1, 16, 8, 8]              32
             ReLU-20             [-1, 16, 8, 8]               0
          Dropout-21             [-1, 16, 8, 8]               0
           Conv2d-22             [-1, 32, 8, 8]           4,608
      BatchNorm2d-23             [-1, 32, 8, 8]              64
             ReLU-24             [-1, 32, 8, 8]               0
          Dropout-25             [-1, 32, 8, 8]               0
           Conv2d-26             [-1, 64, 8, 8]          18,432
      BatchNorm2d-27             [-1, 64, 8, 8]             128
             ReLU-28             [-1, 64, 8, 8]               0
          Dropout-29             [-1, 64, 8, 8]               0
           Conv2d-30             [-1, 16, 8, 8]           1,024
      BatchNorm2d-31             [-1, 16, 8, 8]              32
             ReLU-32             [-1, 16, 8, 8]               0
          Dropout-33             [-1, 16, 8, 8]               0
           Conv2d-34             [-1, 32, 8, 8]           4,608
      BatchNorm2d-35             [-1, 32, 8, 8]              64
             ReLU-36             [-1, 32, 8, 8]               0
          Dropout-37             [-1, 32, 8, 8]               0
           Conv2d-38             [-1, 64, 8, 8]          18,432
      BatchNorm2d-39             [-1, 64, 8, 8]             128
             ReLU-40             [-1, 64, 8, 8]               0
          Dropout-41             [-1, 64, 8, 8]               0
           Conv2d-42             [-1, 32, 6, 6]          18,432
      BatchNorm2d-43             [-1, 32, 6, 6]              64
             ReLU-44             [-1, 32, 6, 6]               0
          Dropout-45             [-1, 32, 6, 6]               0
AdaptiveAvgPool2d-46             [-1, 32, 1, 1]               0
           Linear-47                   [-1, 10]             330
================================================================
Total params: 92,394
Trainable params: 92,394
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 2.36
Params size (MB): 0.35
Estimated Total Size (MB): 2.73
----------------------------------------------------------------
```

### &emsp; **Receptive Field Calculation Table for the final model (CIFAR-10)**

***

This table tracks the Receptive Field (RF) size and the feature map dimensions (Output Size) throughout this model, assuming a $\mathbf{32 \times 32}$ CIFAR10 input image which has 3 input channels.

| **Block** | **Layer** | **Input Size (HxW)** | **Output Size (HxW)** | **Input Channels** | **Output Channels** | **Receptive Field (RF)** | **Details** |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Input** | **Input** | 32x32 | 32x32 | 3 | 3 | **1** | Initial image |
| $\text{B1}$ | **c1\_1** | 32x32 | 32x32 | 3 | 32 | **3** | $k=3, S=1, P=1, D=1$ |
| $\text{B1}$ | **c1\_2** | 32x32 | **16x16** | 32 | 64 | **5** | $k=3, S=2, P=1, D=1$ (Downsampling) |
| $\text{B1}$ | **onexonec1\_1** | 16x16 | 16x16 | 64 | 16 | **5** | $k=1, S=1, P=0, D=1$ (Channel Reduction) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| $\text{B2}$ | **c2\_1** | 16x16 | 16x16 | 16 | 32 | **9** | $k=3, S=1, P=1, D=1$ |
| $\text{B2}$ | **c2\_2 (DW)** | 16x16 | **8x8** | 32 | 32 | **13** | $k=3, S=2, P=1, D=1$ (Depthwise + Downsampling) |
| $\text{B2}$ | **c2\_2 (PW)** | 8x8 | 8x8 | 32 | 16 | **13** | $k=1, S=1, P=0, D=1$ (Pointwise) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| $\text{B3}$ | **c3\_1** | 8x8 | 8x8 | 16 | 32 | **21** | $k=3, S=1, P=1, D=1$ |
| $\text{B3}$ | **c3\_2** | 8x8 | 8x8 | 32 | 64 | **29** | $k=3, S=1, P=1, D=1$ |
| $\text{B3}$ | **onexonec3\_1** | 8x8 | 8x8 | 64 | 16 | **29** | $k=1, S=1, P=0, D=1$ (Channel Reduction) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| $\text{B4}$ | **c4\_1** | 8x8 | 8x8 | 16 | 32 | **37** | $k=3, S=1, P=1, D=1$ |
| $\text{B4}$ | **c4\_2** | 8x8 | 8x8 | 32 | 64 | **45** | $k=3, S=1, P=1, D=1$ |
| $\text{B4}$ | **c5 (Dilated)** | 8x8 | 8x8 | 64 | 32 | **61** | $k=3, S=1, P=1, \mathbf{D=2}$ (RF expanded) |
| **Output** | **GAP** | 8x8 | 1x1 | 32 | 32 | **61** | Global Average Pooling |
| **Output** | **FC** | 1x1 | 1x1 | 32 | 10 | **61** | Classification |


### &emsp; **Training Logs**

<div style="height: 500px; overflow-y: auto; padding: 10px; border: 1px solid #ccc;">
 EPOCH: 1
Epoch=1 Loss=2.0002 Accuracy=19.58: 100%|██████████| 391/391 [00:21<00:00, 18.13it/s]

Test set: Average loss: 0.0156, Accuracy: 2567/10000 (25.67%)

found perfect model!!
Saved best model (acc=25.67%) to best_model.pth

EPOCH: 2
Epoch=2 Loss=1.8868 Accuracy=29.41: 100%|██████████| 391/391 [00:19<00:00, 19.90it/s]

Test set: Average loss: 0.0142, Accuracy: 3362/10000 (33.62%)

found perfect model!!
Saved best model (acc=33.62%) to best_model.pth

EPOCH: 3
Epoch=3 Loss=1.8216 Accuracy=34.99: 100%|██████████| 391/391 [00:20<00:00, 18.63it/s]

Test set: Average loss: 0.0127, Accuracy: 4018/10000 (40.18%)

found perfect model!!
Saved best model (acc=40.18%) to best_model.pth

EPOCH: 4
Epoch=4 Loss=1.6157 Accuracy=40.55: 100%|██████████| 391/391 [00:20<00:00, 19.30it/s]

Test set: Average loss: 0.0115, Accuracy: 4649/10000 (46.49%)

found perfect model!!
Saved best model (acc=46.49%) to best_model.pth

EPOCH: 5
Epoch=5 Loss=1.6717 Accuracy=45.01: 100%|██████████| 391/391 [00:19<00:00, 20.34it/s]

Test set: Average loss: 0.0103, Accuracy: 5224/10000 (52.24%)

found perfect model!!
Saved best model (acc=52.24%) to best_model.pth

EPOCH: 6
Epoch=6 Loss=1.4599 Accuracy=48.86: 100%|██████████| 391/391 [00:20<00:00, 19.19it/s]

Test set: Average loss: 0.0095, Accuracy: 5630/10000 (56.30%)

found perfect model!!
Saved best model (acc=56.30%) to best_model.pth

EPOCH: 7
Epoch=7 Loss=1.2851 Accuracy=52.00: 100%|██████████| 391/391 [00:19<00:00, 20.14it/s]

Test set: Average loss: 0.0090, Accuracy: 5900/10000 (59.00%)

found perfect model!!
Saved best model (acc=59.00%) to best_model.pth

EPOCH: 8
Epoch=8 Loss=1.1936 Accuracy=54.20: 100%|██████████| 391/391 [00:19<00:00, 20.41it/s]

Test set: Average loss: 0.0084, Accuracy: 6213/10000 (62.13%)

found perfect model!!
Saved best model (acc=62.13%) to best_model.pth

EPOCH: 9
Epoch=9 Loss=1.1123 Accuracy=56.31: 100%|██████████| 391/391 [00:20<00:00, 19.18it/s]

Test set: Average loss: 0.0078, Accuracy: 6408/10000 (64.08%)

found perfect model!!
Saved best model (acc=64.08%) to best_model.pth

EPOCH: 10
Epoch=10 Loss=0.9096 Accuracy=58.27: 100%|██████████| 391/391 [00:19<00:00, 20.28it/s]

Test set: Average loss: 0.0077, Accuracy: 6510/10000 (65.10%)

found perfect model!!
Saved best model (acc=65.10%) to best_model.pth

EPOCH: 11
Epoch=11 Loss=1.0461 Accuracy=60.06: 100%|██████████| 391/391 [00:20<00:00, 19.24it/s]

Test set: Average loss: 0.0072, Accuracy: 6691/10000 (66.91%)

found perfect model!!
Saved best model (acc=66.91%) to best_model.pth

EPOCH: 12
Epoch=12 Loss=1.1812 Accuracy=61.73: 100%|██████████| 391/391 [00:19<00:00, 19.97it/s]

Test set: Average loss: 0.0076, Accuracy: 6616/10000 (66.16%)


EPOCH: 13
Epoch=13 Loss=1.1266 Accuracy=62.83: 100%|██████████| 391/391 [00:19<00:00, 19.66it/s]

Test set: Average loss: 0.0067, Accuracy: 7020/10000 (70.20%)

found perfect model!!
Saved best model (acc=70.20%) to best_model.pth

EPOCH: 14
Epoch=14 Loss=0.8683 Accuracy=64.14: 100%|██████████| 391/391 [00:19<00:00, 19.57it/s]

Test set: Average loss: 0.0068, Accuracy: 6928/10000 (69.28%)


EPOCH: 15
Epoch=15 Loss=0.9463 Accuracy=65.06: 100%|██████████| 391/391 [00:18<00:00, 20.69it/s]

Test set: Average loss: 0.0069, Accuracy: 6954/10000 (69.54%)


EPOCH: 16
Epoch=16 Loss=0.8844 Accuracy=65.82: 100%|██████████| 391/391 [00:20<00:00, 19.29it/s]

Test set: Average loss: 0.0064, Accuracy: 7147/10000 (71.47%)

found perfect model!!
Saved best model (acc=71.47%) to best_model.pth

EPOCH: 17
Epoch=17 Loss=0.8517 Accuracy=66.82: 100%|██████████| 391/391 [00:18<00:00, 20.77it/s]

Test set: Average loss: 0.0057, Accuracy: 7427/10000 (74.27%)

found perfect model!!
Saved best model (acc=74.27%) to best_model.pth

EPOCH: 18
Epoch=18 Loss=0.8899 Accuracy=67.74: 100%|██████████| 391/391 [00:19<00:00, 19.71it/s]

Test set: Average loss: 0.0059, Accuracy: 7368/10000 (73.68%)


EPOCH: 19
Epoch=19 Loss=1.0219 Accuracy=68.24: 100%|██████████| 391/391 [00:19<00:00, 19.65it/s]

Test set: Average loss: 0.0056, Accuracy: 7516/10000 (75.16%)

found perfect model!!
Saved best model (acc=75.16%) to best_model.pth

EPOCH: 20
Epoch=20 Loss=0.8869 Accuracy=68.65: 100%|██████████| 391/391 [00:19<00:00, 20.35it/s]

Test set: Average loss: 0.0055, Accuracy: 7555/10000 (75.55%)

found perfect model!!
Saved best model (acc=75.55%) to best_model.pth

EPOCH: 21
Epoch=21 Loss=0.7102 Accuracy=69.56: 100%|██████████| 391/391 [00:21<00:00, 18.33it/s]

Test set: Average loss: 0.0051, Accuracy: 7751/10000 (77.51%)

found perfect model!!
Saved best model (acc=77.51%) to best_model.pth

EPOCH: 22
Epoch=22 Loss=0.6476 Accuracy=69.69: 100%|██████████| 391/391 [00:19<00:00, 19.78it/s]

Test set: Average loss: 0.0052, Accuracy: 7703/10000 (77.03%)


EPOCH: 23
Epoch=23 Loss=0.7444 Accuracy=69.85: 100%|██████████| 391/391 [00:19<00:00, 19.99it/s]

Test set: Average loss: 0.0052, Accuracy: 7694/10000 (76.94%)


EPOCH: 24
Epoch=24 Loss=0.5914 Accuracy=70.49: 100%|██████████| 391/391 [00:20<00:00, 19.30it/s]

Test set: Average loss: 0.0049, Accuracy: 7784/10000 (77.84%)

found perfect model!!
Saved best model (acc=77.84%) to best_model.pth

EPOCH: 25
Epoch=25 Loss=1.0081 Accuracy=70.80: 100%|██████████| 391/391 [00:19<00:00, 20.33it/s]

Test set: Average loss: 0.0051, Accuracy: 7785/10000 (77.85%)

found perfect model!!
Saved best model (acc=77.85%) to best_model.pth

EPOCH: 26
Epoch=26 Loss=0.9620 Accuracy=71.02: 100%|██████████| 391/391 [00:20<00:00, 19.13it/s]

Test set: Average loss: 0.0050, Accuracy: 7813/10000 (78.13%)

found perfect model!!
Saved best model (acc=78.13%) to best_model.pth

EPOCH: 27
Epoch=27 Loss=0.8064 Accuracy=71.74: 100%|██████████| 391/391 [00:19<00:00, 20.09it/s]

Test set: Average loss: 0.0049, Accuracy: 7816/10000 (78.16%)

found perfect model!!
Saved best model (acc=78.16%) to best_model.pth

EPOCH: 28
Epoch=28 Loss=0.6570 Accuracy=71.58: 100%|██████████| 391/391 [00:19<00:00, 19.56it/s]

Test set: Average loss: 0.0050, Accuracy: 7818/10000 (78.18%)

found perfect model!!
Saved best model (acc=78.18%) to best_model.pth

EPOCH: 29
Epoch=29 Loss=0.5593 Accuracy=72.06: 100%|██████████| 391/391 [00:19<00:00, 20.10it/s]

Test set: Average loss: 0.0047, Accuracy: 7939/10000 (79.39%)

found perfect model!!
Saved best model (acc=79.39%) to best_model.pth

EPOCH: 30
Epoch=30 Loss=0.5878 Accuracy=72.30: 100%|██████████| 391/391 [00:18<00:00, 21.09it/s]

Test set: Average loss: 0.0052, Accuracy: 7770/10000 (77.70%)


EPOCH: 31
Epoch=31 Loss=0.9945 Accuracy=72.64: 100%|██████████| 391/391 [00:19<00:00, 20.01it/s]

Test set: Average loss: 0.0048, Accuracy: 7956/10000 (79.56%)

found perfect model!!
Saved best model (acc=79.56%) to best_model.pth

EPOCH: 32
Epoch=32 Loss=0.7216 Accuracy=72.80: 100%|██████████| 391/391 [00:18<00:00, 20.84it/s]

Test set: Average loss: 0.0049, Accuracy: 7849/10000 (78.49%)


EPOCH: 33
Epoch=33 Loss=0.8515 Accuracy=72.90: 100%|██████████| 391/391 [00:19<00:00, 19.56it/s]

Test set: Average loss: 0.0046, Accuracy: 8027/10000 (80.27%)

found perfect model!!
Saved best model (acc=80.27%) to best_model.pth

EPOCH: 34
Epoch=34 Loss=0.9907 Accuracy=73.41: 100%|██████████| 391/391 [00:18<00:00, 21.13it/s]

Test set: Average loss: 0.0043, Accuracy: 8048/10000 (80.48%)

found perfect model!!
Saved best model (acc=80.48%) to best_model.pth

EPOCH: 35
Epoch=35 Loss=0.8681 Accuracy=73.71: 100%|██████████| 391/391 [00:19<00:00, 20.12it/s]

Test set: Average loss: 0.0044, Accuracy: 8070/10000 (80.70%)

found perfect model!!
Saved best model (acc=80.70%) to best_model.pth

EPOCH: 36
Epoch=36 Loss=0.7526 Accuracy=73.68: 100%|██████████| 391/391 [00:18<00:00, 21.29it/s]

Test set: Average loss: 0.0044, Accuracy: 8084/10000 (80.84%)

found perfect model!!
Saved best model (acc=80.84%) to best_model.pth

EPOCH: 37
Epoch=37 Loss=0.6341 Accuracy=73.94: 100%|██████████| 391/391 [00:19<00:00, 19.93it/s]

Test set: Average loss: 0.0042, Accuracy: 8224/10000 (82.24%)

found perfect model!!
Saved best model (acc=82.24%) to best_model.pth

EPOCH: 38
Epoch=38 Loss=0.6523 Accuracy=74.23: 100%|██████████| 391/391 [00:18<00:00, 21.40it/s]

Test set: Average loss: 0.0043, Accuracy: 8151/10000 (81.51%)


EPOCH: 39
Epoch=39 Loss=0.8711 Accuracy=74.48: 100%|██████████| 391/391 [00:19<00:00, 19.58it/s]

Test set: Average loss: 0.0045, Accuracy: 8073/10000 (80.73%)


EPOCH: 40
Epoch=40 Loss=0.8961 Accuracy=74.49: 100%|██████████| 391/391 [00:19<00:00, 20.00it/s]

Test set: Average loss: 0.0043, Accuracy: 8159/10000 (81.59%)


EPOCH: 41
Epoch=41 Loss=0.7181 Accuracy=74.95: 100%|██████████| 391/391 [00:18<00:00, 20.85it/s]

Test set: Average loss: 0.0041, Accuracy: 8205/10000 (82.05%)


EPOCH: 42
Epoch=42 Loss=0.6727 Accuracy=74.71: 100%|██████████| 391/391 [00:20<00:00, 19.04it/s]

Test set: Average loss: 0.0044, Accuracy: 8075/10000 (80.75%)


EPOCH: 43
Epoch=43 Loss=0.5930 Accuracy=75.00: 100%|██████████| 391/391 [00:19<00:00, 20.44it/s]

Test set: Average loss: 0.0044, Accuracy: 8057/10000 (80.57%)


EPOCH: 44
Epoch=44 Loss=0.4716 Accuracy=75.34: 100%|██████████| 391/391 [00:20<00:00, 19.37it/s]

Test set: Average loss: 0.0040, Accuracy: 8220/10000 (82.20%)


EPOCH: 45
Epoch=45 Loss=0.8495 Accuracy=75.66: 100%|██████████| 391/391 [00:20<00:00, 19.38it/s]

Test set: Average loss: 0.0041, Accuracy: 8177/10000 (81.77%)


EPOCH: 46
Epoch=46 Loss=0.5432 Accuracy=75.49: 100%|██████████| 391/391 [00:19<00:00, 19.74it/s]

Test set: Average loss: 0.0040, Accuracy: 8244/10000 (82.44%)

found perfect model!!
Saved best model (acc=82.44%) to best_model.pth

EPOCH: 47
Epoch=47 Loss=0.6235 Accuracy=75.98: 100%|██████████| 391/391 [00:20<00:00, 18.72it/s]

Test set: Average loss: 0.0039, Accuracy: 8286/10000 (82.86%)

found perfect model!!
Saved best model (acc=82.86%) to best_model.pth

EPOCH: 48
Epoch=48 Loss=0.6899 Accuracy=76.06: 100%|██████████| 391/391 [00:19<00:00, 20.01it/s]

Test set: Average loss: 0.0039, Accuracy: 8262/10000 (82.62%)


EPOCH: 49
Epoch=49 Loss=0.5012 Accuracy=76.14: 100%|██████████| 391/391 [00:20<00:00, 18.98it/s]

Test set: Average loss: 0.0038, Accuracy: 8365/10000 (83.65%)

found perfect model!!
Saved best model (acc=83.65%) to best_model.pth

EPOCH: 50
Epoch=50 Loss=0.6600 Accuracy=76.17: 100%|██████████| 391/391 [00:19<00:00, 19.77it/s]

Test set: Average loss: 0.0039, Accuracy: 8315/10000 (83.15%)


EPOCH: 51
Epoch=51 Loss=0.8176 Accuracy=76.33: 100%|██████████| 391/391 [00:19<00:00, 19.89it/s]

Test set: Average loss: 0.0039, Accuracy: 8245/10000 (82.45%)


EPOCH: 52
Epoch=52 Loss=0.6320 Accuracy=76.42: 100%|██████████| 391/391 [00:20<00:00, 18.71it/s]

Test set: Average loss: 0.0037, Accuracy: 8331/10000 (83.31%)


EPOCH: 53
Epoch=53 Loss=0.6481 Accuracy=76.89: 100%|██████████| 391/391 [00:19<00:00, 20.05it/s]

Test set: Average loss: 0.0038, Accuracy: 8339/10000 (83.39%)


EPOCH: 54
Epoch=54 Loss=0.6629 Accuracy=76.91: 100%|██████████| 391/391 [00:20<00:00, 18.75it/s]

Test set: Average loss: 0.0037, Accuracy: 8382/10000 (83.82%)

found perfect model!!
Saved best model (acc=83.82%) to best_model.pth

EPOCH: 55
Epoch=55 Loss=0.9218 Accuracy=77.15: 100%|██████████| 391/391 [00:20<00:00, 19.24it/s]

Test set: Average loss: 0.0038, Accuracy: 8346/10000 (83.46%)


EPOCH: 56
Epoch=56 Loss=0.8345 Accuracy=76.73: 100%|██████████| 391/391 [00:19<00:00, 20.00it/s]

Test set: Average loss: 0.0037, Accuracy: 8364/10000 (83.64%)


EPOCH: 57
Epoch=57 Loss=0.6126 Accuracy=77.25: 100%|██████████| 391/391 [00:20<00:00, 18.64it/s]

Test set: Average loss: 0.0037, Accuracy: 8367/10000 (83.67%)


EPOCH: 58
Epoch=58 Loss=0.7614 Accuracy=77.31: 100%|██████████| 391/391 [00:19<00:00, 20.13it/s]

Test set: Average loss: 0.0036, Accuracy: 8427/10000 (84.27%)

found perfect model!!
Saved best model (acc=84.27%) to best_model.pth

EPOCH: 59
Epoch=59 Loss=0.6854 Accuracy=77.40: 100%|██████████| 391/391 [00:20<00:00, 18.75it/s]

Test set: Average loss: 0.0037, Accuracy: 8387/10000 (83.87%)


EPOCH: 60
Epoch=60 Loss=0.5720 Accuracy=77.46: 100%|██████████| 391/391 [00:19<00:00, 19.92it/s]

Test set: Average loss: 0.0036, Accuracy: 8439/10000 (84.39%)

found perfect model!!
Saved best model (acc=84.39%) to best_model.pth

EPOCH: 61
Epoch=61 Loss=0.6017 Accuracy=77.24: 100%|██████████| 391/391 [00:19<00:00, 20.50it/s]

Test set: Average loss: 0.0036, Accuracy: 8429/10000 (84.29%)


EPOCH: 62
Epoch=62 Loss=0.6483 Accuracy=77.79: 100%|██████████| 391/391 [00:20<00:00, 19.23it/s]

Test set: Average loss: 0.0036, Accuracy: 8430/10000 (84.30%)


EPOCH: 63
Epoch=63 Loss=0.3760 Accuracy=77.80: 100%|██████████| 391/391 [00:19<00:00, 20.40it/s]

Test set: Average loss: 0.0035, Accuracy: 8461/10000 (84.61%)

found perfect model!!
Saved best model (acc=84.61%) to best_model.pth

EPOCH: 64
Epoch=64 Loss=0.6661 Accuracy=77.96: 100%|██████████| 391/391 [00:20<00:00, 18.97it/s]

Test set: Average loss: 0.0035, Accuracy: 8447/10000 (84.47%)


EPOCH: 65
Epoch=65 Loss=0.6809 Accuracy=77.92: 100%|██████████| 391/391 [00:19<00:00, 20.20it/s]

Test set: Average loss: 0.0035, Accuracy: 8469/10000 (84.69%)

found perfect model!!
Saved best model (acc=84.69%) to best_model.pth

EPOCH: 66
Epoch=66 Loss=0.5377 Accuracy=78.33: 100%|██████████| 391/391 [00:20<00:00, 18.90it/s]

Test set: Average loss: 0.0035, Accuracy: 8491/10000 (84.91%)

found perfect model!!
Saved best model (acc=84.91%) to best_model.pth

EPOCH: 67
Epoch=67 Loss=0.4668 Accuracy=78.14: 100%|██████████| 391/391 [00:20<00:00, 19.28it/s]

Test set: Average loss: 0.0035, Accuracy: 8472/10000 (84.72%)


EPOCH: 68
Epoch=68 Loss=0.4888 Accuracy=78.24: 100%|██████████| 391/391 [00:19<00:00, 20.36it/s]

Test set: Average loss: 0.0035, Accuracy: 8480/10000 (84.80%)


EPOCH: 69
Epoch=69 Loss=0.4606 Accuracy=78.29: 100%|██████████| 391/391 [00:20<00:00, 19.22it/s]

Test set: Average loss: 0.0035, Accuracy: 8513/10000 (85.13%)

found perfect model!!
Saved best model (acc=85.13%) to best_model.pth
 
</div>

### &emsp; **Training Graphs**
<table width="100%">
  <tr>
    <td width="50%" align="center">
      <img src="images/training_loss.png" alt="Graph 1 Title" style="width: 100%; max-width: 400px;"/>
      <br>
      **Figure 1: Training Loss**
    </td>
    <td width="50%" align="center">
      <img src="images/test_loss.png" alt="Graph 2 Title" style="width: 100%; max-width: 400px;"/>
      <br>
      **Figure 2: Testing Loss**
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <img src="images/training_accuracy.png" alt="Graph 3 Title" style="width: 100%; max-width: 400px;"/>
      <br>
      **Figure 3: Training Accuracy**
    </td>
    <td width="50%" align="center">
      <img src="images/test_accuracy.png" alt="Graph 4 Title" style="width: 100%; max-width: 400px;"/>
      <br>
      **Figure 4: Testing Accuracy**
    </td>
  </tr>
</table>
