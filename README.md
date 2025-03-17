# HPOseq
**HPOseq: A Deep Ensemble Model for Predicting the Protein-Phenotype Relationships Based on Protein Sequences**
<div align="center">
  <img src="https://github.com/LabBioMedCoder/HPOseq/blob/main/HPOseq_structure.png" width="800px" height="600px">
</div>
The framework of HPOseq.

# Dependencies
* keras==2.9.0
* numpy==1.22.3
* pandas==2.0.3
* scikit_learn==1.3.2
* scipy==1.11.4
* tensorflow_gpu==2.9.0
* torch==1.7.1

# Dataset
In the data folder, we provide all the necessary data for the model. Specifically, we offer one fold of a five-fold cross-validation, which includes both the training set (train.json) and the test set (test.json). Additionally, we have provided processed feature sequences in the form of .rar files. Simply download and merge these files during runtime, and then can be directly used  for our model.
 
# Resources
* creats_genelist.py: Download the gene-HPO relationship file from the HPO database and generate genelist.txt and gene_hpo.json based on the file
* Authentic labels.py: Add the ancestor nodes of the current HPO annotation relation to the relation after processing.
* intra.py: Prediction using intra-sequence features
* inter.py: Prediction using inter-sequence features
* Ensemble_module.py: train and test the model

# Code
**Python implementation files for HPOseq**

     1. inter.py - This file is the code for the intra-sequence feature sub-model, which is run to obtain the probability distribution of the existence of an association for each disease phenotype.

     2. intra.py - This file is the code for the inter-sequence feature sub-model, which can be run to obtain the probability distribution of the existence of an association for each disease phenotype.

     3. Ensemble_module.py - This file is the code for the submodel result fusion module, which consists of a fully-connected neural network and a mask matrix for fusing prediction results under the intra- and inter-sequence feature submodels to generate more accurate disease phenotype association scores.


# Result Generation
After the model is run, the "result.csv" will be generated and saved in the current folder.

# Run
* The required packages can be installed by running `pip install -r requirements.txt`.
* The first step is to run `python intra.py` to obtain the in-sequence feature sub-model predictions.
* The second step runs `python inter.py` to obtain the inter-sequence feature sub-model predictions.
* The third step runs `python Ensemble_module.py`, which fuses the two independent submodel predictions to generate the final prediction.
