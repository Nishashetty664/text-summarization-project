Text Summarization Project

Project Overview

This project aims to develop a text summarization model using a dataset from the CNN/DailyMail website. The goal is to create a model that can generate concise and informative summaries of news articles. The project follows a structured approach, including data collection, preprocessing, model training, and evaluation.

### Data Collection

The dataset was sourced from the CNN/DailyMail website due to its rich collection of news articles spanning various topics and genres. This dataset was chosen for its diversity and relevance to current events, ensuring that the model trained on this data would be able to generate summaries that are informative and up-to-date. The dataset size was 11,460 records. 

Data Preprocessing

During the data preprocessing phase, the dataset underwent several steps to clean and prepare it for model training:

Tokenization: The text was tokenized into individual words using NLTK's word_tokenize method.
Stopwords Removal: Common English stopwords were removed using the NLTK stopwords corpus.

Punctuation Removal: Punctuation marks were removed from the text using the translate method with string.punctuation.

Lowercasing: All text was converted to lowercase to ensure consistency.

Stemming: Words were stemmed using the Porter stemming algorithm to reduce them to their base or root form.

Lemmatization: Lemmatization was applied to further reduce words to their base form using the WordNet lemmatizer.

The cleaned and preprocessed text data was saved to a new CSV file (preprocessed_data.csv), containing the 'id', 'cleaned_article', and 'cleaned_highlights' columns.

Dataset Splitting

The preprocessed dataset was split into three sets: train, validate, and test, with proportions of 80%, 10%, and 10%, respectively. This splitting was done using the train_test_split method from scikit-learn.

Model Training

Utilized the T5 transformer model for text summarization because T5 is known for its effectiveness in generating high-quality summaries, making it a suitable choice for this project.
Implemented training using the native PyTorch method.Used the AdamW optimizer with a learning rate of 1e-5 and weight decay of 0.01 for regularization.Trained the model for 7 epochs with a batch size of 8.
Result::Training loss=1.84.

Evaluation

For the evaluation stage, the model's performance will be assessed using the ROUGE metric, which is commonly used for evaluating text summarization models. 
