# FLOWER IMAGE CLASSIFIER PROJECT

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#overview">Overview</a></li>
    <li><a href="#dependencies">Dependencies</a></li>
    <li>
      <a href="#part-1-developing-classifier-in-jupyter-notebook">
        Part 1: Developing classifier in Jupyter Notebook
      </a>
    </li>
    <li>
      <a href="#part-2-building-the-command-line-application">Part 2: Building the command line application</a>
      <ul>
        <li><a href="#application-usage">Application Usage</a></li>
      </ul>
    </li>
  </ol>
</details>

## Overview
The purpose of this project is to utilize a pre-trained model from [PyTorch](https://pytorch.org) to build, train, and implement an image classifier that recognizes different species of flowers using transfer learning technique.

The project is broken down into two parts: developing the classifier in Jupyter Notebook and then building the command line application for the classifier.

Additionally, please keep all files intact to avoid error.

&nbsp;
## Dependencies
```
numpy>=1.22.1
torch>=1.12.1
torchvision>=0.13.1
matplotlib>=3.5.3
pillow>=9.0.0
```
The following command line will help install the required dependencies conveniently:
```
pip -r requirements.txt
```

&nbsp;
## Part 1: Developing classifier in Jupyter Notebook
The image classifier is built and trained step by step with a thorough instruction in [Image Classifier Project.ipynb](Image%20Classifier%20Project.ipynb).

&nbsp;
## Part 2: Building the command line application
This is the main part of the project. The code written in Part 1 is reused as part of the application for the classifier. 

The application consists of two main functions: 
- Construct a new neural network on a dataset with user-defined architecture, train the initialized model, and saves it as checkpoint.
- Load the trained model from the checkpoint file and deploy it on user-chosen images.

### Application Usage:

1. **Model construction and training:**
    ```
    py train.py [-h] 
                [--save_dir [SAVE_DIR]] 
                [--arch {vgg11,vgg13,vgg16,vgg19}] 
                [--learning_rate ALPHA] 
                [--hidden_units [HIDDEN_UNITS ...]] 
                [--epochs EPOCHS] 
                [--batch_size BATCH_SIZE] 
                [--drop_p P] 
                [--gpu] 
                data_dir
    ```
* Positional arguments:
    * `data_dir`: Input directory which contains two sub-folders of dataset: _train_ and _val_.

* Optional arguments:
    * `-h, --help`: Show help message and exit
    * `--save_dir [SAVE_DIR]`: Input directory where information of trained model will be saved. No argument means saving at the current working directory.
    * `--arch {vgg11,vgg13,vgg16,vgg19}`: Choose a VGG model architecture. Default is vgg19.
    * `--learning_rate ALPHA`: Set learning rate value. Default is 0.001.
    * `--hidden_units [HIDDEN_UNITS ...]`: Input multiple integers separated by a single space to design the hidden layers for the classification part of the model.
    * `--epochs EPOCHS`: Set the number of epochs. Default is 10.
    * `--batch_size BATCH_SIZE`: Set the size of each batch. Default is 32.
    * `--drop_p P`: Set probability for dropout regularization. Default is 0.
    * `--gpu`: Allow the program to use GPU to train the model. No arguments needed.

&nbsp;

2. **Model loading and deployment:**
    ```
    py predict.py [-h] 
                  [--topk K] 
                  [--category_names JSON] 
                  [--gpu] 
                  img_path 
                  checkpoint
    ```
* Positional arguments:
    * `img_path`: Input the path to the image which will be predicted by the model.
    * `checkpoint`: Input the path to the file which contains trained model's information.

* Optional arguments:
    * `-h, --help`: Show help message and exit
    * `--topk K`: Input the number of top classes to be displayed. Default is 3.
    * `--category_names JSON`: Input the path to the JSON file which is a mapping of categories to real name of flowers.
    * `--gpu`: Allow the program to use GPU to perform prediction. No arguments needed.