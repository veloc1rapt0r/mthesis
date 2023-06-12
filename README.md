# Residual Neural Networks for Medical Image Processing: A Case Study on Liver Cancer Diagnostics

This repository contains code examples related to Oleh Bakumenko's master thesis conducted at the Faculty of Mathematics, University of Duisburg-Essen. 
The thesis focuses on the implementation and evaluation of various classification and segmentation models for the analysis of liver CT scans.
The presented models in the thesis include: AlexNet, ResNet18, ResNet34, ResNet50, ResNet152, VGG19, and U-Net. Please refer to the PDF directory and the References section in the thesis for more information about these models.

### Folder Structure and Descriptions:
- code: This folder contains the scripts for model definitions and performance comparison.
- images: It stores the images illustrated in the PDF.
- images_gallery: This folder contains saved images for the "Section 6: Gallery" with examples of classification and segmentation.
- logs: It contains information sources for performance comparison, saved as CSV tables. Each log file represents a specific model with the following subscripts:
  - _val: Validation loss and accuracy after each epoch.
  - _test: Test loss and accuracy at the end of the training.
  - _train: Train loss saved for each batch number in each epoch.
  - _runtime: Runtime of each epoch in seconds.
- utility: This folder consists of utility files for creating datasets, logging routines, calculation of train/val/test losses, performance metrics, plotting routines, generalized train loop, and more.

The one can start with /code/plots_and_graphs.ipynb notebook, as it contains all the classification and segmentation performance comparison, described in details in Section 5 of the thesis.
The notebook uses pandas dataframes, such that the reader can easily reach needed information using names of the model as keywords. 
For the paragraph "Robustness with respect to initialization." the "_init" subscript will indicate the normal-initialized version of the model; a model without the subscript indicates the default PyTorch uniform initialization.

### Colab
There is an opportunity to start the project without using the local python interpreter and cloning the Git repository. 
Google Colab, also known as "Colaboratory," provides a browser-based Python environment where you can write and execute code effortlessly. It requires no configuration, offers free access to GPUs, and comes with pre-installed packages.
To use Colab, you need to sign in with your Google account to access its computational resources. 
GitHub copy of this project was needed, so it was created in:
https://github.com/veloc1rapt0r/mthesis.git
To use colab the one needs:
1. Go to https://research.google.com/colaboratory/
2. In the menu above choose File-> Open notebook -> GitHub
3. Enter a GitHub URL https://github.com/veloc1rapt0r/mthesis.git
4. Choose the "mthesis" repository and the "main" branch.
5. Select the desired file from the list, for example as "code/plots_and_graphs.ipynb."
6. After the file is opened in a new window, navigate through directories using change directory (cd) or list files (ls) commands or paste as separate cells:
  
  !git clone https://git.uni-due.de/skolbaku/residual-neural-networks-for-medical-image-processing-a-case-study-in-liver-cancer-diagnostics.git
  
  cd residual-neural-networks-for-medical-image-processing-a-case-study-in-liver-cancer-diagnostics/code
  
  ls

7. Make sure that the file you want to open is displayed in the list of files. Failure to do so may result in certain generalized functions not working correctly, for example as importing logs.  
