# Psaraftis-Souranis Symeon Informatics and Computer Engineering Diploma Thesis Code
Thesis Title: Usage of Machine Learning Techniques for Medical Image Analysis in Order to  Assist in the Diagnosis of COVID-19 Infection</br></br>
[@simospsaraftis](https://github.com/simospsaraftis)</br></br>

## Description

This repository contains code developed for my diploma thesis: "Usage of Machine Learning Techniques for Medical Image Analysis in Order to Assist in the Diagnosis of COVID-19 Infection."
</br></br>

## Classification Task

The code trains convolutional neural networks (CNNs) to classify chest CT scans based on the presence of COVID-19 lesions.
</br></br>

### Models Trained

The following three models were trained:

- ResNet50
- InceptionV3
- Xception
- DenseNet121
</br></br>

### Dataset Used for Classification Task

For this thesis the dataset created by [1] was employed. It can be downloaded from here: [COVID-CT-MD: COVID-19 Computed Tomography (CT) Scan Dataset Applicable in Machine Learning and Deep Learning](https://figshare.com/collections/COVID-CT-MD_COVID-19_Computed_Tomography_CT_Scan_Dataset_Applicable_in_Machine_Learning_and_Deep_Learning/5129081)

The dataset should be added here:

- `segmentation/data/images` Add your input CT scan images here.
- `segmentation/data/masks` Add the corresponding CT scan masks here.
</br></br>

## Segmentation Task

The code trains convolutional neural networks (CNNs) to segment COVID-19-related lesions in chest CT scans, supporting automated analysis and diagnosis.
</br></br>

### Models Trained

The following three models were trained utilizing code found in [2]:

- U-Net
- U-Net with VGG16 as backbone
- U-Net with DenseNet121 as backbone
</br></br>

### Dataset Used for Segmentation Task

For this thesis the dataset created by [3] was employed. It can be downloaded from here: [COVID-19 CT Lung and Infection Segmentation Dataset](https://zenodo.org/records/3757476)

The dataset should be added here:

- `segmentation/data/images` Add your input CT scan images here.
- `segmentation/data/masks` Add the corresponding CT scan masks here.
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
</br></br>

## References

[1] Afshar, P., Heidarian, S., Enshaei, N., Naderkhani, F., Rafiee, M. J., Oikonomou, A., Babaki Fard, F., Samimi, K., Plataniotis, K. N., & Mohammadi, A. (2021). *COVID-CT-MD, COVID-19 computed tomography scan dataset applicable in machine learning and deep learning*. Scientific Data, 8(1), 121. https://doi.org/10.1038/s41597-021-00900-3

[2] Pavel Iakubovskii. *Segmentation Models*. GitHub repository, 2019. Available at: [https://github.com/qubvel/segmentation_models](https://github.com/qubvel/segmentation_models)

[3] Ma, J., Wang, Y., An, X., Ge, C., Yu, Z., Chen, J., Zhu, Q., Dong, G., He, J., He, Z., Cao, T., Zhu, Y., Nie, Z., & Yang, X. (2021). Towards Data-Efficient Learning: A Benchmark for COVID-19 CT Lung and Infection Segmentation. *Medical Physics*, 48(3), 1197–1210. https://doi.org/10.1002/mp.14676
