ejemplo: https://github.com/S-parera/RL-chess_aidl/blob/main/README.md

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

## Experiment setup

**To be explained**

# Datasets

# Recommender system models:

https://towardsdatascience.com/factorization-machines-for-item-recommendation-with-implicit-feedback-data-5655a7c749db#:~:text=Factorization%20Machines%20(FM)%20are%20generic,regression%2C%20classification%2C%20and%20ranking.

https://towardsdatascience.com/introduction-to-recommender-systems-2-deep-neural-network-based-recommendation-systems-4e4484e64746

https://stats.stackexchange.com/questions/108901/difference-between-factorization-machines-and-matrix-factorization

## Factorization machines

Factorization Machines (FM) are generic supervised learning models that map arbitrary real-valued features into a low-dimensional latent factor space [2]. FM models represent user-item interactions as tuples of real-valued feature vectors and numeric target variables (similar to a regression or classification models). The base features will be binary vectors of user and item indicators, such that each training sample has exactly two non-zero entries corresponding to the given user/item combination.

![image](https://user-images.githubusercontent.com/93614965/156051487-d82a03fd-c020-46fc-890d-6dff592614ac.png)


## Factorization machines with GCN

## Factorization machines with GCN (with an attention layer)

  

# Results



Citation

[1] https://www.researchgate.net/figure/Schematic-representation-of-the-drug-discovery-process-The-two-main-phases-discovery_fig2_335215729

[2] https://towardsdatascience.com/factorization-machines-for-item-recommendation-with-implicit-feedback-data-5655a7c749db#:~:text=Factorization%20Machines%20(FM)%20are%20generic,regression%2C%20classification%2C%20and%20ranking.



Luo, Y., Zhao, X., Zhou, J., Yang, J., Zhang, Y., Kuang, W., Peng, J., Chen, L. & Zeng, J. A network integration approach for drug-target interaction prediction and computational drug repositioning from heterogeneous information. Nature Communications 8, (2017).
