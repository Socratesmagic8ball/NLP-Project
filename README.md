# Filtering Hate Speech in LLM Prompts: An NLP Approach

## Project Overview
This repository explores the detection of toxic language and media bias within Large Language Model (LLM) interactions. By training a model on benchmarked media bias datasets, the project aims to identify harmful biases—such as hate speech and racial bias—specifically focusing on nuanced contexts like sarcasm where positive words may be used negatively.

## Group Members
This project was developed as a collaborative effort by:
* **Varnika K.**
* **Kaavya Lakshmi**
* **V Venkatesh**
* **Chacko C**

## Methodology
The project implements a multi-stage NLP pipeline to bridge the gap between media bias detection and real-world AI prompt filtering:

1.  **Model Selection**: Fine-tuning the **BERT (Bidirectional Encoder Representations from Transformers)** base model for sequence classification.
2.  **Training Data**: Utilized the **MBIB (Media Bias Identification Benchmark)** dataset, focusing on the `hate_speech` and `racial_bias` subsets.
3.  **Cross-Domain Application**: Applied the trained model to the `10k_prompts_ranked` dataset from HuggingFace to evaluate how well media bias detection transfers to human-AI conversational prompts.
4.  **Sentiment Integration**: Integrated **VADER (Valence Aware Dictionary and sEntiment Reasoner)** to detect "positive signal words" used in biased contexts, aiding in the identification of ironic or sarcastic hate speech.

## Key Results & Performance
The fine-tuned BERT model achieved high reliability in identifying problematic prompts:
* **Accuracy**: 86.2%
* **Weighted F1-Score**: 0.863
* **Precision**: 0.865
* **Analysis**: The pipeline identified **118 potential ironic/sarcastic candidates** where strong positive words (e.g., *excellent, successful, proud*) were utilized within a classified biased context.

## Future Implications
The ultimate goal of this work is to improve the safety layers of AI models, particularly in mental health applications, by providing a "candid" understanding of how users communicate and perceive the world during their interactions with AI.

## Requirements
* **Frameworks**: PyTorch, Transformers (HuggingFace)
* **Libraries**: `datasets`, `scikit-learn`, `nltk`, `pandas`, `tqdm`
