# Text Summarization Project

## Project Overview:

This project aims to develop a text summarization model using a dataset from the CNN/DailyMail website. The goal is to create a model that can generate concise and informative summaries of news articles. The project follows a structured approach, including data collection, preprocessing, model training, and evaluation.

### Data Collection

The dataset was sourced from the CNN/DailyMail website due to its rich collection of news articles spanning various topics and genres. This dataset was chosen for its diversity and relevance to current events, ensuring that the model trained on this data would be able to generate summaries that are informative and up-to-date. The dataset size was 11,490 records. 

### Data Preprocessing

During the data preprocessing phase, the dataset underwent several steps to clean and prepare it for model training:

#### Tokenization: 
The text was tokenized into individual words using NLTK's word_tokenize method.
#### Stopwords Removal:
Common English stopwords were removed using the NLTK stopwords corpus.
#### Punctuation Removal: 
Punctuation marks were removed from the text using the translate method with string.punctuation.
#### Lowercasing:
All text was converted to lowercase to ensure consistency.
#### Stemming: 
Words were stemmed using the Porter stemming algorithm to reduce them to their base or root form.
#### Lemmatization:
Lemmatization was applied to further reduce words to their base form using the WordNet lemmatizer.

The cleaned and preprocessed text data was saved to a new CSV file (preprocessed_data.csv), containing the 'id', 'cleaned_article', and 'cleaned_highlights' columns.

### Dataset Splitting

The preprocessed dataset was split into three sets: train, validate, and test, with proportions of 80%, 10%, and 10%, respectively. This splitting was done using the train_test_split method from scikit-learn.
[train_dataset](https://drive.google.com/file/d/1GmQ9fkA93uDFb2-tMghMDspJ6FMWtKEI/view?usp=drive_link)


### Model Training

Utilized the T5 transformer model for text summarization because T5 is known for its effectiveness in generating high-quality summaries, making it a suitable choice for this project.
Implemented training using the native PyTorch method.Used the AdamW optimizer with a learning rate of 1e-5 and weight decay of 0.01 for regularization.
Result::Training Loss: 1.09                                                                                                Validation Loss: 0.94

### Evaluation

###### Method Used:

Utilized the ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metric to evaluate the quality of the generated summaries.Calculated ROUGE scores for ROUGE-1, ROUGE-2, and ROUGE-L.
Implemented a custom function calculate_rouge to calculate ROUGE scores for each summary against its reference summary.
Achieved an average ROUGE-1 F1 score of 0.35, an average ROUGE-2 F1 score of 0.15, and an average ROUGE-L F1 score of 0.25 on the validation set.
# Hyperparameter Tuning:

Experiment with different hyperparameters, such as the learning rate, batch size, and number of training epochs, to further improve the model's performance.
after the hyperparameter tunung the performance increased from 0.32 to 0.50 rough1 score.

## Interface Development:

Implemented an interactive text summarization interface using Gradio, leveraging a fine-tuned T5 model for abstractive summarization and the TextRank algorithm for extractive summarization, with PDF processing capabilities using PyMuPDF (fitz).
### future work 
working on report and deployment of model
