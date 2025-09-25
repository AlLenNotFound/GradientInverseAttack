# Code for the NeurIPS 2024 submission: </br></br>"DAGER: Extracting Text from Gradients with Language Model Priors"
## Prerequisites
- Install Anaconda. 
- Create the conda environment:<br>

> conda env create -f environment.yml -n dager

- Enable the created environment:<br>

> conda activate dager

- Create necessary folders

> mkdir -p models models_cache

- Download the required files from [Megadrive](https://mega.nz/folder/6E0ilJRA#ALIOb7a8ugfKOSmhCSEW6A) and store the fine-tuned GPT-2 model and LoRA models in the `\models` folder

## Setting up HuggingFace (optional)
We use HuggingFace for obtaining all datasets and models. However, the LLaMa-2 (7B) and ECHR datasets rely on a specific setup.

For installing the ECHR dataset follow the next steps:
> wget https://huggingface.co/datasets/glnmario/ECHR/resolve/main/ECHR_Dataset.csv

> mkdir ./models\_cache/datasets--glnmario--ECHR

> mv ECHR\_Dataset.csv ./models\_cache/datasets--glnmario--ECHR/

For using the LLaMa-2 (7B) model, you need to create a HuggingFace profile and request access to the model from Meta. Then, create an API token that allows read access to relevant repositories. Export this token into an environment variable as shown below:
> export HF\_TOKEN=<huggingface\_api\_token>

*NOTE*: if you have any issues connecting to HuggingFace we recommend using the  `huggingface-cli` as so:
> huggingface-cli download \<model\> --cache-dir ./models\_cache

## Running TAG/LAMP

We provide a modified verson of TAG and LAMP inside the `\lamp` folder that we used to run the baselines for GPT-2. For experiments on BERT, we recommend using the [original code](https://github.com/eth-sri/lamp). Further instructions can be found inside the `README.md` inside the respective repositories.

## Rank tolerance adjustment
As general guidance, for each experiment we define a rank tolerance parameter *RANK_TOL* to deal with numerical instabilities. This parameter has to be tweaked depeding on the batch size, with any value within an order of magnitude of the "true value" resulting in equivalent behaviour. Throughout our experiments, as rule of thumb we followed the below guidelines:
- *RANK_TOL* = $10^{-7}$ for batch sizes 1/2
- *RANK_TOL* = $10^{-8}$ for batch sizes 4-16
- *RANK_TOL* = $10^{-9}$ for batch sizes 32-128

We set a default rank tolerance of `None` for automatic inference (as specified in the NumPy documentation).

## DAGER Experiments (Tables 1, 2, 3, 5, 11)
### Parameters
- *DATASET* - the dataset to use. Must be one of **cola**, **sst2**, **rotten\_tomatoes**.
- *BATCH\_SIZE* - how many sentences we have in a batch.

### Commands
To run GPT-2:
> ./scripts/gpt2.sh DATASET BATCH\_SIZE <--rank\_tol RANK\_TOL>

To run GPT-2 Large:
> ./scripts/gpt2-large.sh DATASET BATCH\_SIZE <--rank\_tol RANK\_TOL>

To run GPT-2 Finetuned:
> ./scripts/gpt2-ft.sh DATASET BATCH\_SIZE <--rank\_tol RANK\_TOL>

To run GPT-2 on the next-token prediction task:
> ./scripts/gpt2.sh DATASET BATCH\_SIZE <--rank\_tol RANK\_TOL> --task next\_token\_pred

To run GPT-2 using the Frobenius norm loss:
> ./scripts/gpt2.sh DATASET BATCH\_SIZE <--rank\_tol RANK\_TOL> --loss mse

To run GPT-2 using a ReLU activation:
> ./scripts/gpt2.sh DATASET BATCH\_SIZE <--rank\_tol RANK\_TOL> --hidden_act relu

To run LLaMa-2:
> ./scripts/llama.sh DATASET BATCH\_SIZE <--rank\_tol RANK\_TOL>

To run BERT (with all heuristics enabled):
> ./scripts/bert.sh DATASET BATCH\_SIZE <--rank\_tol RANK\_TOL>

To run BERT (with heuristics disabled):
> ./scripts/bert.sh DATASET BATCH\_SIZE <--rank\_tol RANK\_TOL> --l1\_filter all --l2\_filter overlap

To run DAGER on GPT-2 under LoRA finetuning (by default run at rank $r=256$):
> ./scripts/lora.sh DATASET BATCH\_SIZE <--rank\_tol RANK\_TOL>

*NOTE*: for the experiment on LLaMa-2 on Rotten Tomatoes for batch size 128, it is recommended to use the following command for speed up and numerical stability:
> ./llama.sh rotten\_tomatoes 128 --rank\_tol 1e-11 --l1\_span\_thresh 1e-4 --l2\_span_thresh 5e-10

Furthermore, for the experiment on long sequences using the `glnmario/ECHR` dataset, it is recommended to also set a low rank tolerance as well as a lower span threshold (we used RANK\_TOL=$10^{-9}$, l1\_span\_thresh=$10^{-4}$)

## DAGER under FedAvg (Table 4)
By default, the algorithm is run on the Rotten Tomatoes dataset with a batch size of 16.

### Parameters
- *AVG_EPOCHS* - How many training iterations of the FedAvg algorithm are run.
- *B_MINI* - The minibatch size.
- *LEARNING_RATE* - The learning rate for the single SGD step.


### Commands
To run DAGER on GPT-2 with the FedAvg algorithm:

> ./scripts/fed_avg.sh AVG_EPOCHS B_MINI LEARNING_RATE 

## Ablation study on best-effort reconstruction (Table 4)
We completed this study through logging our results in [Neptune](https://neptune.ai/). We can retrieve the effect of the rank thresholding by setting the scores of all runs with `logs\batch_tokens > <model_embedding_dim>` to 0, and recomputing the aggregate statistics.

## Ablation study on filter thresholding (Figure 2)
To run a test on the effect of the span check filter across various thresholds, ranging from $10^{-7}$ to 1, run:
> python ./token_filtering.py

## Ablation study on rank threshold (Figure 3)
This study relates to finding the optimal rank threshold and showcasing what happens when it is too high or low. This study **requires** the ECHR dataset - see how to set it up in the `Setting up HuggingFace` section.

### Parameters
- *RANK\_CUTOFF* - the amount we lower the maximum rank by with respect to the embedding dimension, after which we apply the cutoff. For example, for RANK\_CUTOFF=20, and an embedding dimension of 768, the maximum rank becomes 748.

### Command
> ./scripts/dec\_feasibility.sh RANK\_CUTOFF

## Ablation study on encoder heuristics (Figure 4)
In this study we how DAGER on encoders performs with and without heuristics across different sequence lengths on batch size 4.

### Parameters

- *END\_INPUT* - at what sequence length the search should finish

### Commands
To run the study on BERT with all heuristics:
> ./scripts/enc\_feasibility.sh 4 END\_INPUT

To run the study on BERT with no heuristics:
> ./scripts/enc\_feasibility\_no\_heuristics.sh 4 END\_INPUT

## Fine-tuning GPT-2 with and without defended gradients

In order to fine-tune GPT-2, you need `scikit-learn` as an additional prerequisite. Simply run either:

> pip install scikit-learn

or

> conda install -c conda-forge scikit-learn

and follow the instructions below.

### Parameters
- *DATASET* - the dataset to use. Must be one of **cola**, **sst2**, **rotten_tomatoes**.
- *SIGMA* - the amount of Gaussian noise with which to train e.g **0.001**. To train without defense set to **0.0**.
- *NUM\_EPOCHS* - for how many epochs to train e.g **2**.

### Commands

- To train your own network:<br>
> python3 train.py --dataset DATASET --batch\_size 32 --noise SIGMA --num\_epochs NUM\_EPOCHS --save\_every 100 --model\_path MODEL

The models are stored under `finetune/<TRAINING_METHOD>_<EPOCH>`, where *TRAINING_METHOD* can be either `lora` or `full`, and the *EPOCH* is the corresponding checkpoint
