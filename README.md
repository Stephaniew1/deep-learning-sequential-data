# Deep Learning for Sequential Data

End-to-end NLP pipeline for 6-class question classification on the TREC 
dataset, built across three notebooks: recurrent architectures implemented 
from scratch in raw PyTorch, then fine-tuned BERT for comparison.

## Project context

Originally developed as an individual project during a Monash deep learning 
course (FIT3181), then refactored and extended for portfolio use. Initial 
dataset loaders and the evaluation harness were provided as course 
scaffolding. All model implementations, comparisons, and analysis are my 
own work.

## What's in here

**01_pipeline_overview.ipynb** — Fundamentals in RNNs. Dataset preparation 
for the TREC 6-class question classification task (categories: ABBR, ENTY, 
DESC, HUM, LOC, NUM), tokenisation, and the evaluation framework used 
across the project.

**02_rnn_gru_lstm_from_scratch.ipynb** — Manual RNN forward pass 
implemented directly with raw PyTorch tensor operations, without using 
`nn.RNN`. Extended into a configurable `BaseRNN` class supporting vanilla 
RNN, GRU, and LSTM variants, with different pooling strategies (last 
hidden state, mean pooling, max pooling) compared head-to-head.

**03_bert_finetuning.ipynb** — Fine-tuning `bert-base-uncased` from 
HuggingFace Transformers for 6-class sequence classification. Includes 
tokenisation, attention masking, training loop with gradient accumulation, 
and evaluation against the from-scratch baselines from Notebook 02.

## Tech stack

- **Notebooks 01 and 02:** Python, PyTorch, NumPy, Scikit-learn
- **Notebook 03:** Python, PyTorch, HuggingFace Transformers, BERT
- Developed in Google Colab for GPU access

## How to run

Open the notebooks in Google Colab (recommended) or Jupyter. Run them in 
numerical order. The TREC dataset is loaded inline in Notebook 01.

For Notebook 03, you will need a HuggingFace account and the 
`transformers` library installed. GPU is strongly recommended for the 
BERT fine-tuning step.

## Why this project

The goal was to understand recurrent architectures from first principles 
before reaching for high-level abstractions, then compare those baselines 
against a fine-tuned transformer on the same task. The manual RNN forced 
me to work through backpropagation through time and tensor reshaping the 
hard way, which made the eventual move to GRU, LSTM, and BERT feel earned 
rather than magical.

## License

MIT