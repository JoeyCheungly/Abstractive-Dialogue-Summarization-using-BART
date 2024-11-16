# Abstractive Dialogue Summarization using BART

## Overview
This project focuses on automating this process by using pre-trained large language model, BART, for dialogue summarization. BART is a powerful pre-trained model used for summarizaing structured text like news articles and journals (Lewis et al., 2019). The aim of this project is to fine-tune the model to better handle conversational data and to improve the accuracy and coherence of dialogue summaries. 


## File Descriptions
Here is a brief overview of the files included in this project:

- **data/samsum-train.csv**: Training dataset
- **data/samsum-validation.csv**: Validation dataset
- **data/samsum-test.csv**: Testing dataset
- **EDA.ipynb**: Exploratory Data Analysis
- **BART.py**: BART model implementation for dialogue summarization 
- **BART_test.py**: Testing of BART model 
- **HFModel.ipynb**: Testing results of uploaded Hugging Face Model 
- **RNN_summarization.ipynb**: RNN implementation for dialogue summarization 
- **README.md**: README file 

## Installation
1. Set Up the Environments for BART and RNN Implementations:
- For BART implementation:
`pip install -r requirements_BART.txt`

- For RNN implementation:
`pip install -r requirements_RNN.txt`

2. Activate BART environment and run the files. For instance,
- Run Python files: `python BART.py`
- Run ipynb files by starting jupyter notebook interface: `jupyter notebook`

