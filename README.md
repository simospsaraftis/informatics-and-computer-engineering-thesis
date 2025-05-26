# Psaraftis-Souranis Symeon Informatics and Computer Engineering Diploma Thesis Code
Thesis Title: Usage of Machine Learning Techniques for Medical Image Analysis in Order to  Assist in the Diagnosis of COVID-19 Infection</br></br>
[@simospsaraftis](https://github.com/simospsaraftis)</br></br>

## Description

This repository contains code developed for my diploma thesis: "Usage of Machine Learning Techniques for Medical Image Analysis in Order to Assist in the Diagnosis of COVID-19 Infection."

The code trains convolutional neural networks (CNNs) to segment COVID-19-related lesions in chest CT scans, supporting automated analysis and diagnosis.

## Models Trained

The following three models were trained utilizing code found in [1]:

- U-Net
- U-Net with VGG16 as backbone
- U-Net with DenseNet121 as backbone
</br></br>

## Installation

Clone the repository:

```
git clone https://github.com/simospsaraftis/informatics-and-computer-engineering-thesis.git
cd informatics-and-computer-engineering-thesis
```

Install the required packages:

```
pip install -r requirements.txt
```

- `data/images` Add your input CT scan images here.
- `data/masks` Add the corresponding CT scan masks here.
- `models/` Trained models will be saved here.

</br></br>
## References

[1] Pavel Iakubovskii. *Segmentation Models*. GitHub repository, 2019. Available at: [https://github.com/qubvel/segmentation_models](https://github.com/qubvel/segmentation_models)
