# Metastasis-glmGCN
Presentation
--------
'Distant metastasis identification based on optimized graph representation of gene interaction patterns'  
In this study, we proposed a graph convolutional network embedded with a graph learning module, named glmGCN, to predict the distant metastasis of cancer.

Dataset
------
Cervical squamous cell carcinoma and endocervical adenocarcinoma(CESC)  
Stomach adenocarcinoma(STAD)  
Pancreatic adenocarcinoma(PAAD)   
Bladder Urothelial Carcinoma(BLCA)  

Protein-Protein Interaction network: https://pan.baidu.com/s/1x8y6PtAmM_HFp2kfyAM3gA 【u8z9】


Version
--------
Python   3.6.12  
CUDA     10.0.130  
Tensorflow-gpu 1.14.0  

Usage
--------
Take PAAD as an example:    
input_PAAD.csv is a gene expression data file, while PPI_PAAD.npy represents protein-protein interaction relationships. We run train_glmGCN.py to get the training and verification results. models_glmGCN.py is the file used to build the model and layers_glmGCN.py is the implementation file of graph learning and graph convolution operation. inits.py and utils.py contain many functions that can be manipulated by other files.

