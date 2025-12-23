# Cat vs Dog Image Classification (PyTorch)

## Project Overview

This project implements a **Cat vs Dog image classification system** using **PyTorch** and **transfer learning**. The goal is to build a complete and reproducible training and inference pipeline for binary image classification.

A pretrained **ResNet-50** model is used as the backbone network. Its original classification head is replaced with a **custom classifier**, which is trained on a labeled cat and dog dataset. During training, the backbone is frozen and only the newly defined classifier layers are optimized. The training process selects and saves the best-performing model based on validation performance.

The dataset used in this project is provided by Udacity and can be downloaded from:

```
https://s3.amazonaws.com/content.udacity-data.com/nd089/Cat_Dog_data.zip
```

After training, the model is evaluated on a held-out test set. The trained model achieves **approximately 98% accuracy on the test dataset**, demonstrating the effectiveness of transfer learning for this task.
