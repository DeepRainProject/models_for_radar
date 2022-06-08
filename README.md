# Deep learning models for  generation of precipitation maps based on NWP

In this repository you can find the code of different deep learning techniques to produce rain maps based on the physical simulations model (or numerical weather predictions NWP). The algorithms double the resolution of the precipitation predictions and correct bias contained in the original forecast with a lead time of 3h.

# Notebooks


## preprocessing.ipynb

The preprocessing routine loads the original COSMO-DE and radar data and assign them to the correspondent time point. It also converts the input data to z-score and creates the train, validation and test sets. It write the datasets in TFRecords format.

## evaluate.ipynb

The evaluate file loads each one of the generated models, saves and produces the respective predictions for the test set. The respective predictions are evaluated using different metrics and saved as a dictionary.

## plot.ipynb

The plot routine is in charge of loading the metrics of the models and plot the respective results.

# Python scripts

The models were trained in a HPC system using MPI to distribute task between processors. Each of the following model scripts were submitted 20 times with random initializations in a different CPU.

## Baseline model

Basic deconvolutional model used as a baseline model for comparison.

##  U-Net model

Adaptation of the U-Net architecture (Ronnenberger et al. 2015) used in the literature in super resolution (Serifi et al. 2021) and nowcasting of precipitation data (Ayzel et al. 2020). Used as a comparison point for new models. 

## Deconv1L model

Model that generates one set of 32 feature maps based on applying deconvolution kernels to the input data. A last convolution layer calculates the rain amount combining the high resolution feature maps.

## Deconv3L model

Inspired on the expansive path of the U-Net architecture, applies a max pooling layer followed by two deconvolution operations until getting the desired output resolution. A final convolutional layer calculates the rain. 

## CGAN model

Using the Deconv3L as a generator, the CGAN model complements the back propagation process of the loss with the probability of fooling an extra model (the discriminator). 
