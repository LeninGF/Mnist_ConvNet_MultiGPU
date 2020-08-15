## Training Mnist with Multiple GPU
This is a small script to test the training of a ConvNet model using multi gpu. It is reported that there are some problems
when saving the h5 file. It seems that these problems are not present in models like vgg-X but it seems to appear in models
with different building blocks (nasnet, resnet, resnext, densenet).

This script also serves as a fast start up to test anaconda environment and use of Hardware Resources (aka GPU!)

### Tensorflow requirements:
These scripts have been produced and tested with Tensorflow v1.13.1. Changes may be required for adapting the code to 
newer versions of tensorflow. Anyway, we provide a yaml file to clone our working environment and fast testing. However,
we advice that the environment contains other python libraries and uses at most 5 GB of disk space.

### Cloning anaconda environment file
From base environment in anaconda use:

    conda env create -f tf_gpu_cuda_100.yaml
    
### Scripts Provided:

* **main.py** -> This script downloads Mnist

* **load_and_test.py** -> This script aims to load the saved models from the h5 file and evaluate them to confirm that
trained model was sucessfully saved

#### If this was of use leave a comment and share. Happy coding.
