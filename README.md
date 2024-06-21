Text Summarization Project

Project Overview

This project aims to develop a text summarization model using a dataset from the CNN/DailyMail website. The goal is to create a model that can generate concise and informative summaries of news articles. The project follows a structured approach, including data collection, preprocessing, model training, and evaluation.

### Data Collection

The dataset was sourced from the CNN/DailyMail website due to its rich collection of news articles spanning various topics and genres. This dataset was chosen for its diversity and relevance to current events, ensuring that the model trained on this data would be able to generate summaries that are informative and up-to-date. The dataset size was 11,460 records. 

*Data Preprocessing*

During the data preprocessing phase, the dataset underwent several steps to clean and prepare it for model training:

Tokenization: The text was tokenized into individual words using NLTK's word_tokenize method.

Stopwords Removal: Common English stopwords were removed using the NLTK stopwords corpus.

Punctuation Removal: Punctuation marks were removed from the text using the translate method with string.punctuation.

Lowercasing: All text was converted to lowercase to ensure consistency.

Stemming: Words were stemmed using the Porter stemming algorithm to reduce them to their base or root form.

Lemmatization: Lemmatization was applied to further reduce words to their base form using the WordNet lemmatizer.

The cleaned and preprocessed text data was saved to a new CSV file (preprocessed_data.csv), containing the 'id', 'cleaned_article', and 'cleaned_highlights' columns 

Splitting the Dataset

After preprocessing, the dataset was split into training, validation, and test datasets. The code for this process is provided in the file preprocessed_split file.This ensures that the model can be trained, validated, and tested on distinct subsets of the data to evaluate its performance accurately.

Model Training

For the training phase, the pre-trained T5 model from Hugging Face was selected and fine-tuned using the training dataset. The training process yielded a training loss of 2.4627, indicating the model's performance on the training data. The code for model training is available in the file fine_tune.file

Evaluation

For the evaluation stage, the model's performance will be assessed using the ROUGE metric, which is commonly used for evaluating text summarization models. 
