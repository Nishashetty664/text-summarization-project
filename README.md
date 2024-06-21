####Text Summarization Project

####Project Overview

This project aims to develop a text summarization model using a dataset from the CNN/DailyMail website. The goal is to create a model that can generate concise and informative summaries of news articles. The project follows a structured approach, including data collection, preprocessing, model training, and evaluation.

### Data Collection

The dataset was sourced from the CNN/DailyMail website due to its rich collection of news articles spanning various topics and genres. This dataset was chosen for its diversity and relevance to current events, ensuring that the model trained on this data would be able to generate summaries that are informative and up-to-date. The dataset size was 11,460 records. 

Preprocessing

The dataset has undergone several preprocessing steps to clean and normalize the text data. These steps include data cleaning, normalization, tokenization, and the removal of stop words. 

Splitting the Dataset

After preprocessing, the dataset was split into training, validation, and test datasets. The code for this process is provided in the file preprocessed_split file.This ensures that the model can be trained, validated, and tested on distinct subsets of the data to evaluate its performance accurately.

Model Training

For the training phase, the pre-trained T5 model from Hugging Face was selected and fine-tuned using the training dataset. The training process yielded a training loss of 2.4627, indicating the model's performance on the training data. The code for model training is available in the file fine_tune.file

Evaluation

For the evaluation stage, the model's performance will be assessed using the ROUGE metric, which is commonly used for evaluating text summarization models. 
