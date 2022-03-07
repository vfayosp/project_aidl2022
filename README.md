# Table of contents
- [Table of contents](#table-of-contents)
- [Getting started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Project structure](#project-structure)
- [Drug recommender system](#drug-recommender-system)
    - [Team members:](#team-members)
    - [Advisor:](#advisor)
    - [Framework:](#framework)
- [Introduction and motivation](#introduction-and-motivation)
  - [Recommender systems:](#recommender-systems)
  - [Drug discovery](#drug-discovery)
  - [Drug repurposing](#drug-repurposing)
- [Project Goals](#project-goals)
  - [Environment setup](#environment-setup)
- [Datasets](#datasets)
- [Recommender system models:](#recommender-system-models)
  - [Factorization machines](#factorization-machines)
  - [Factorization machines with GCN](#factorization-machines-with-gcn)
  - [Factorization machines with GCN (with an attention layer)](#factorization-machines-with-gcn-with-an-attention-layer)
- [Metrics, evaluations](#metrics-evaluations)
- [Experiment setup, model approaches and results.](#experiment-setup-model-approaches-and-results)
  - [Protein-drug](#protein-drug)
- [Results](#results)
# Getting started
## Prerequisites
To install and run de project the use Miniconda 3 is advised: 
````
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
````
After the install of Miniconda the setting of a new environment is adviced also:
````
conda create --name project python=3.9
conda activate project
````
A requeriments file is also provided in order to set up all the necesary packages to run the main program:
````
pip install -r requirements.txt
````

## Installation
In order to launch the program it is only needed to clone the git repo:
````
git clone https://github.com/vfayosp/project_aidl2022.git
````

## Project structure
The project follows this structure:
````
.
├── data                             
    ├── cleaned_data
        ├── df_drug_disease.csv          # data without repetitions
        └── ...
    ├── original_data
        ├── disease.txt                  # list of diseases
        ├── mat_drug_disease.txt         # matrix of interactions
        └── ...
    ├── prepared_data
        ├── df_train_drug_disease.csv    # sample data used for train
        ├── df_test_drug_disease.csv     # sample data used for test
        └── ...
    ├── prepared_data_context
        ├── df_train_drug_disease2.csv    # sample data used for train with context
        ├── df_test_drug_disease2.csv     # sample data used for test with context
        └── ...
├── trained_models                   # Trained models ready to predict
├── utils                            # Auxiliar functions used by the main
├── main.py                          # Main program
├── requeriments.txt                 # Requirements file
├── LICENSE
└── README.md
````
# Drug recommender system 
Final project for the postgraduate degree in Artificial Intelligence with Deep Learning


### Team members:
Victor Fayos, Jose Mérida, Lucía Pau, Alba Puy
### Advisor:
Paula Gomez

### Framework:
Pytorch

# Introduction and motivation

## Recommender systems:

A recommendation system is a information filtering system that helps to predict the "preference" between and user and an item. This is used in a wide variety of areas mainly  including music, films, books and products. 

From an user-product point of  view (which is the widely used approach) , recommender system are usually based on :

* **Collaborative** filtering: Builds a model based on user behavior and interactions as well as similar decissions of other users.
* **Content-based** filtering: Uses the characteristics of the item itself to find similar properties between items and recommender them.

![image](https://user-images.githubusercontent.com/93614965/156049316-3cb3a26f-478c-4fbf-8247-96046d5d66cf.png)


However, it has the potential to be used in several fields, such as the medical field. There are already existing studies about its applications to:

* Side effects
* Drug repositioning
* Drug-drug interaction
* Treatment recommendations.







## Drug discovery

Drug discovery is the process of identifying  potencial new medicines. This process start with **target identification and validation**, after that, several clinical studies and clinical developments are performed before it is registrated and obtains the approval to be released.

![image](https://user-images.githubusercontent.com/93614965/156042866-4300912b-a0b0-48b0-8f6d-fb837f6dc7be.png)
[1]


Research productivity is known as a time consuming, expensive path, which most of the times end up with very low sucess drug candidates. On this basis, recommender systems have the potential to become useful tools for this process. 



## Drug repurposing

Drug repurposing consist on identifying new targets for known drugs. In this regards, using an existing drug can save a lot of steps in drug discovery and development proccess.

Among other advantages, it is demonstrated to:

* Reduce R&D costs
* Decrease drug development timeline
* Make clinical trials easier. 
 
Making clinical tries easier is one of the main paths of improvement, since it requires testing medical approaches in people. An offline evaluation, since this project, allows researchers in a medical environment to predict future preferences, which could potentially reduce needed testing in both people and animals.



# Project Goals

* Apply what we learn in the post-graduate program in the medicine field
* Build two recommender systems for drug candidates:

  *   **Protein-drug recommender**: Potential use to link a known target selected (first step in drug discovery process) for a specific disease to an existing drug.
  *   **Drug-disease recommender**: envisioned in the field of drug repurposing or drug reposisioning (exploring the use of drugs already in place to treat other diseases).

## Environment setup
For running the following things shoul be described:

 . Criterion
 · Optimizers
 · Learning rate
 · Batch size		
 · Number of epochs
 . TopK
 . Embed dimm
 
For the implementation of our code we used Adam as optimizer, with a learning rate of 0.01. The batch size used was 256, Embeding dimension was set to 64, we trained for 150 epochs using 10 as topk. As criterion we used BCEWithLogitsLoss(reduction='mean').


# Data

## Data manipulation

The data was manipulated by the following notebooks:

**1 - Data explotation and cleaning**: Some of the drugs, proteins and diseases were duplicated or triplicated so they were represented multiple times in the data. We kept just one instance of each if the rows or columns were identical and we deleted all the representtions if there was a slightly difference. The sparsity of our data was calculated there.

**R script**: We obliged (separately) each dataset to each row to have at least two interactions and to each column to at least have one. We did that not just for the data we cleaned but also computing the transpose to drug_protein in order tdo have protein_drug. 

**2 - Data Preparation for the model with context**: We obliged each dataset to each row to have at least two interactions and to each column to at least have one. It has been done in a way that the drugs, diseases and proteins kept are the same in all three dataframes.  Also, we divided the data in train and test by selecting randonly one of the interactions of each row and keeping it as test saving the others for train.

Train and test dataframes have been generated with the *random split* approach [5]. This is based on selecting only one random item of each user for testing.
Although this scheme has been changed to use the last interaction (from time point of view, leave one last item), our dataset and the approach of our analysis do not allow for a the temporal approach.

## Datasets

data/directory contains the following directories:

- **original_data**: where the uncleaned data is stored with the following datasets    
- **cleaned_data**: where the cleaned is stored and contains the following datasets (computed using 1 - Data explotation and cleaning)
- **prepared_data**: where train and test are already divided for each dataset (computed using the R script) 
- **prepared_data_context**: where train and test are already divided for each dataset in order to use them for the computation with context (computed using 2 - Data Preparation for the model with context)

# Recommender system models:

## Factorization machines

Factorization Machines (FM) are generic supervised learning models that map arbitrary real-valued features into a low-dimensional latent factor space [2]. It enhances the linear regresion model by incorporating the second-order feature interactions.

FM models represent user-item interactions as tuples of real-valued feature vectors and numeric target variables (similar to a regression or classification models). The base features will be binary vectors of user and item indicators, such that each training sample has exactly two non-zero entries corresponding to the given user/item combination.

![image](https://user-images.githubusercontent.com/93614965/156051487-d82a03fd-c020-46fc-890d-6dff592614ac.png)



## Factorization machines with GCN

Factorization machines assume that each sample is independent and cannot exploit interactions between samples, it is only focused on the features. However, in some applications the interaction between samples is also useful, as in recommendation systems.  

Graph Convolutional Networks (GCN) allow to capture the correlation between nodes by using a convolution operation. This is performed by agreggating information from the neighbors' information when making predictios so the interaction between nodes is also encoded.

## Factorization machines with GCN (with an attention layer)

Factorization machines only incorporate secon-order interactions between features, so high-order interactions are not taken into account. FM can model all feature interaction with the same weight but not all interactions are useful and predictive. Therefore, an attention layer to learn the importance of different feature interactions is included [4]
  
 Attention-based layers permit different contribution from parts to be compressed in a single representation.
 
## Context

In the models explained above the data used consist only on the interactions between items, to those models it is usefull add context to the training. The context takes into account the features of each pair of items interaction. This provides to the model extra information that can be used to better recommend the items.

# Metrics, evaluations

* **Hit ratio**: Is is simply the number of correct items that were present in the recommendation list (of lenght TopK). If the topk is increased the hit ratio increases, but it must be a reasonable value.
* **NDCG** (Normalized Discounted Cumulative Gain) :

The cumulative gain is sthe sum of gains up to the K position in the recommendation list but does not take into account the order. To penalize the gain by its rank, the DCG is introduced, being IDCG the score for the most ideal ranking.

![image](https://user-images.githubusercontent.com/93614965/156443858-3d50d206-c5f9-4b76-b6b7-a5c30bd22243.png)

Then, the NDCG is the DCG normalized by the IDCG so the value is between 0 and 1.

![image](https://user-images.githubusercontent.com/93614965/156443885-32be2840-b528-49a2-b5c1-f88740ffb243.png)


# Experiment setup, model approaches and results.

## Protein-drug

The models explained above where applied to the Protein-Drug dataset to buld up a recommender system able to recommend drugs starting from a target.

* First aproach: Factorization machines model computing embeddings just as a linear layer.
* Second approach: Factorization machines computing embedings with GCN, to take advantage of the interaction between nodes.
* Finally, in order to capture interaction higher than second order, the attention layer was applied.

The attention layer did not show major improvements. The obtained results are showed below:


|          Model         | HR  | NDGC |
|:------------------------:|:-------:|:--------:|
|    FM@10    |  0.2264 |0.2105|
|    FM with GCN@10           |   0.6792  |0.5258|
| FM with GCN and attetion layer@10 |   0.6509  |0.5259|


![image](https://user-images.githubusercontent.com/93614965/156647510-8e419cfe-16e2-447e-b1f5-3ee842114bd2.png)





# Results



Citation

[1] https://www.researchgate.net/figure/Schematic-representation-of-the-drug-discovery-process-The-two-main-phases-discovery_fig2_335215729

[2] https://towardsdatascience.com/factorization-machines-for-item-recommendation-with-implicit-feedback-data-5655a7c749db#:~:text=Factorization%20Machines%20(FM)%20are%20generic,regression%2C%20classification%2C%20and%20ranking.

[3] Attentional Factorization Machines:
Learning the Weight of Feature Interactions via Attention Networks∗

[4] DEEP RELATIONAL FACTORIZATION MACHINES

[5] Exploring Data Splitting Strategies for the Evaluation
of Recommendation Models

Luo, Y., Zhao, X., Zhou, J., Yang, J., Zhang, Y., Kuang, W., Peng, J., Chen, L. & Zeng, J. A network integration approach for drug-target interaction prediction and computational drug repositioning from heterogeneous information. Nature Communications 8, (2017).

https://towardsdatascience.com/factorization-machines-for-item-recommendation-with-implicit-feedback-data-5655a7c749db#:~:text=Factorization%20Machines%20(FM)%20are%20generic,regression%2C%20classification%2C%20and%20ranking.

https://towardsdatascience.com/introduction-to-recommender-systems-2-deep-neural-network-based-recommendation-systems-4e4484e64746

https://stats.stackexchange.com/questions/108901/difference-between-factorization-machines-and-matrix-factorization
