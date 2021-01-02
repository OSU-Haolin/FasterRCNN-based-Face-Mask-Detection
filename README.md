# Face-Mask-Detection
人脸口罩检测（中文）/ Face mask detection (English)

## Environment

We recommend to use `anaconda` to create a python3 environment to manage the pytorch-GPU environment.   
You can use the following commands to configure your environment:
```shell
conda create -n {your environment name} python=3.7
```
Then `anaconda` will solve the dependencies automatically for you. (Make sure you have successfully installed the NVIDIA driver.)

Then you need to install following dependencies in your conda environment:  
```
pytorch-GPU or pytorch > 1.0  
python-opencv > 3.0  
torchvision
numpy
```

## Getting Started
1. Download the AIZOO Face Mask Detection Dataset (or you can use your own dataset, make sure in same format)
([link](https://github.com/AIZOOTech/FaceMaskDetection)) 
and copy the dataset `AIZOO` into the root folder as `/Face-Mask-Detection/AIZOO`.
The files in `AIZOO` are as follow:
   ```shell script
   /AIZOO/
    ├── train
    ├── val
    └── readme.md
    ```   
2.Process the Dataset by running 'dataset_copy.py'. In our demo, we use 800 images for training and 500 images for validation.  
Then processed `train` folder and `val` folder will show in the root folder as follow:    
    ```shell script  
   /ROOT/  
    ├── train  
    │   ├── Annotations  
    │   ├── JPEGImages  
    │   └── train.txt  
    └── val  
        ├── Annotations  
        ├── JPEGImages  
        └── val.txt  
    ```
    note: you can select how many images you want use for training and validation by change 'dataset_copy.py'.  
