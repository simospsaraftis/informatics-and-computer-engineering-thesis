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

Your dataset should be added here:

- `segmentation/data/images` Add your input CT scan images here.
- `segmentation/data/masks` Add the corresponding CT scan masks here.
</br></br>

## Dataset Used

For this thesis the dataset created by [2] was employed. It can be downloaded from here: [COVID-19 CT Lung and Infection Segmentation Dataset](https://zenodo.org/records/3757476)

</br></br>
## References

[1] Pavel Iakubovskii. *Segmentation Models*. GitHub repository, 2019. Available at: [https://github.com/qubvel/segmentation_models](https://github.com/qubvel/segmentation_models)

[2] Ma, J., Wang, Y., An, X., Ge, C., Yu, Z., Chen, J., Zhu, Q., Dong, G., He, J., He, Z., Cao, T., Zhu, Y., Nie, Z., & Yang, X. (2021). Towards Data-Efficient Learning: A Benchmark for COVID-19 CT Lung and Infection Segmentation. *Medical Physics*, 48(3), 1197–1210. https://doi.org/10.1002/mp.14676
