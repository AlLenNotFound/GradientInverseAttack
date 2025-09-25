# LAMP: Extracting Text from Gradients with <br/> Language Model Priors 

We adapt the code from the original LAMP repository and list the relevant commands below:

## Prerequisites
- Install Anaconda. 
- Create the conda environment:<br>

> conda env create -f environment.yml

- Enable the created environment:<br>

> conda activate lamp

## Baseline experiments (Table 1)

### Parameters
- *DATASET* - the dataset to use. Must be one of **cola**, **sst2**, **rotten_tomatoes**, **glnmario/ECHR**.
- *BATCH\_SIZE* - the batch size to use.

### Commands
- To run the experiment on LAMP with cosine loss:<br>
> ./lamp_cos.sh gpt2 DATASET BATCH\_SIZE
- To run the experiment on LAMP with L1+L2 loss:<br>
> ./lamp_l1l2.sh gpt2 DATASET BATCH\_SIZE
- To run the experiment on TAG:<br>
> ./tag.sh gpt2 DATASET BATCH\_SIZE


