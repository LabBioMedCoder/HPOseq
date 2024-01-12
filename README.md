# HPOseq
**HPOseq: A Deep Ensemble Model for Predicting the Protein-Phenotype Relationships Based on Protein Sequences**
<div align="center">
  <img src="https://github.com/LabBioMedCoder/HPOseq/blob/main/HPOseq_structure.png" width="800px" height="300px">
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
In the data folder, we provide the data needed for the model.

# Resources
* creats_genelist.py: Download the gene-HPO relationship file from the HPO database and generate genelist.txt and gene_hpo.json based on the file
* Authentic labels.py: Add the ancestor nodes of the current HPO annotation relation to the relation after processing.
* intra.py: Prediction using intra-sequence features
* inter.py: Prediction using inter-sequence features
* Ensemble_module.py: train and test the model


# Result Generation
After the model is run, the "result.csv" will be generated and saved in the current folder.

# Run
* python Ensemble_module.py
