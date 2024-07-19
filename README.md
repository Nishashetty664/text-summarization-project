# Text Summarization Project

## Project Overview:

This project aims to develop a text summarization model using a dataset from the CNN/DailyMail website. The goal is to create a model that can generate concise and informative summaries of news articles. The project follows a structured approach, including data collection, preprocessing, model training, and evaluation.

### Data Collection

#### Dataset Description

Initial Dataset: The main dataset was initially saved as dataset.csv, containing 70,000 records.
[master_dataset](https://drive.google.com/file/d/1jjldQUckT-HQWkgLgwtXu4ClKm5LE0NV/view?usp=drive_link)

Reason for Reduction: Due to constraints such as processing power, memory limitations, and the need for efficient handling, I reduced the dataset size to 11,490 records.

##### Reduced Dataset: 
You can find the link to the reduced dataset [reduced_data](https://drive.google.com/file/d/1qGaVyolBbiYAyZLcJ65R4qZ5HIMrB2b3/view?usp=drive_link).

### Data Preprocessing

During the data preprocessing phase, the dataset underwent several steps to clean and prepare it for model training:

#### Tokenization: 
The text was tokenized into individual words using NLTK's word_tokenize method.

#### Stopwords Removal:
Common English stopwords were removed using the NLTK stopwords corpus.

#### Lowercasing:
All text was converted to lowercase to ensure consistency.
Words were stemmed using the Porter stemming algorithm to reduce them to their base or root form.

#### Lemmatization:
Lemmatization was applied to further reduce words to their base form using the WordNet lemmatizer.

The cleaned and preprocessed text data was saved to a new CSV file (demo.csv), containing the 'id', 'cleaned_article', and 'cleaned_highlights' columns.
[demo.csv](https://colab.research.google.com/drive/1GVC787vYicnkk4xXrc9_zBl1VjMXBeXE?usp=drive_link).

### Dataset Splitting

The preprocessed dataset was split into three sets: train, validate, and test, with proportions of 80%, 10%, and 10%, respectively. This splitting was done using the train_test_split method from scikit-learn.
[train_dataset](https://drive.google.com/file/d/1GmQ9fkA93uDFb2-tMghMDspJ6FMWtKEI/view?usp=drive_link)
[validate_dataset](https://drive.google.com/file/d/1tKxOxz-22GiiLqpuorJHxx0ZWP7GxMXM/view?usp=drive_link).
[test_dataset](https://drive.google.com/file/d/1qGaVyolBbiYAyZLcJ65R4qZ5HIMrB2b3/view?usp=drive_link).

#### Text summarization can generally be done in two main ways:

•	1. Extractive Summarization:

•	2. Abstractive Summarization:


### Model Training

Utilized the T5 transformer model for text summarization because T5 is known for its effectiveness in generating high-quality summaries, making it a suitable choice for this project.
Implemented training using the native PyTorch method.Used the AdamW optimizer with a learning rate of 1e-5 and weight decay of 0.01 for regularization.
Result::Training Loss: 1.05                                                                                                Validation Loss: 0.96

### Evaluation

###### Method Used:

Utilized the ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metric to evaluate the quality of the generated summaries.Calculated ROUGE scores for ROUGE-1, ROUGE-2, and ROUGE-L.
Implemented a custom function calculate_rouge to calculate ROUGE scores for each summary against its reference summary.
Significant enhancements in model performance were observed following rigorous hyperparameter tuning.
Initial ROUGE-1, ROUGE-2, and ROUGE-L scores stood at 0.322, 0.142, and 0.237 respectively.
Post tuning, these metrics improved to 0.505, 0.339, and 0.437, demonstrating substantial progress in summarization accuracy.

# Hyperparameter Tuning:

Experiment with different hyperparameters, such as the learning rate, batch size, and number of training epochs, to further improve the model's performance.
after the hyperparameter tunung the performance increased from 0.32 to 0.50 rough1 score.
trained model and tokenizer:[fine-tuning](https://drive.google.com/drive/folders/1yrKurkP2rmjW48tUTzR7LrgssDyndRq9?usp=drive_link)


### 	Extractive text summarization

•	Objective: The objective is to develop a system that generates concise summaries by selecting key sentences from the source text, and preserving original wording and context.

•	Used textrank algorithm 

#### RESULT:

o	Precision: The average precision score was approximately 0.119.

o	Recall: The average recall score was approximately 0.813.

o	F1 Score: The average F1 score was approximately 0.203.

## Interface Development:


Implemented an interactive text summarization interface using Gradio, leveraging a fine-tuned T5 model for abstractive summarization and the TextRank algorithm for extractive summarization, with PDF processing capabilities using PyMuPDF (fitz).

###  results

![image](https://github.com/user-attachments/assets/b41e1626-7108-4ebd-be59-652c3f84f282)
![image](https://github.com/user-attachments/assets/8b4523a8-786c-4134-bfbb-db73abb9e97b)


 
 ### Conclusion
 
o	The development of an automated text summarization system using advanced NLP techniques has proven effective and efficient. 

o	The Gradio interface provided a user-friendly platform for summarizing long texts and PDF documents.

o	Using the T5-small model for abstractive summarization and the TextRank algorithm for extractive summarization, the system produced concise, relevant summaries that retained essential information. 

o	This project highlighted the potential of NLP technologies to enhance information retrieval and readability, benefiting various business applications and beyond.


                                                ![image](https://github.com/user-attachments/assets/e3cd79de-c303-4210-b0bb-90d992da9c80)




