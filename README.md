# DATA7903 Capstone Project 2

## Overview
This project focuses on automating this process by using pre-trained large language model, BART, for dialogue summarization. BART is a powerful pre-trained model used for summarizaing structured text like news articles and journals (Lewis et al., 2019). The aim of this project is to fine-tune the model to better handle conversational data and to improve the accuracy and coherence of dialogue summaries. 


## File Descriptions
Here is a brief overview of the files included in this project:

- **data/samsum-train.csv**: Training dataset
- **data/samsum-validation.csv**: Validation dataset
- **data/samsum-test.csv**: Testing dataset
- **EDA.ipynb**: Exploratory Data Analysis
- **BART.py**: BART model implementation for dialogue summarization 
- **Testing.ipynb**: Testing of BART model in Hugging Face
- **RNN_summarization.ipynb**: RNN implementation for dialogue summarization 


## Installation
1. Unzip the `s4763354.zip` file:  
   `unzip s4763354.zip`  
   `cd s4763354`

2. Installed the dependencies for two different environments (BART and RNN implementations):  
   `pip install -r requirements_BART.txt`
   `pip install -r requirements_RNN.txt`
   
## Running the Models
1. First Stage - Run the rankers and rerankers using the same identifiers (A, B, or C) to retrieve documents  
   `python BART.py`  
   `python rerankerA.py`

    Do the same for other rankers and rerankers.

    Note: Reranker A corresponds to Ranker A, and so forth for other identifiers.

4. Second Stage - Run the RAG models to generate answer:  
   `python RAGA.py` 

    Do the same for other RAG models. 

5. Evaluation - Evaluate the retreival models:  
   `python MyRetEval.py`
   
   Evaluate the RAG models:  
   `python MyRAGEval.py`

6. Please find the results above in the output/ folder as JSON files. 