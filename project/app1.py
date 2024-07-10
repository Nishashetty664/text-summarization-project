from transformers import T5ForConditionalGeneration, T5Tokenizer
import gradio as gr
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF for PDF processing

# Load the fine-tuned T5 model for abstractive summarization
model_path = 'fine_tuning'
abstractive_model = T5ForConditionalGeneration.from_pretrained(model_path)
abstractive_tokenizer = T5Tokenizer.from_pretrained(model_path)

# Function for abstractive summarization using the T5 model
def abstractive_summarize(text):
    inputs = abstractive_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = abstractive_model.generate(inputs, max_length=150, num_beams=4, early_stopping=True)
    return abstractive_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Function for extractive summarization using the TextRank algorithm
def extractive_summarize(text):
    sentences = text.split(".")
    
    # Ensure there are enough sentences for summarization
    if len(sentences) < 2:
        return "Not enough sentences to summarize."
    
    # Use CountVectorizer to transform sentences into vectors
    vectorizer = CountVectorizer().fit_transform(sentences)
    vectors = vectorizer.toarray()
    
    # Check if vectors are not empty and have the expected shape
    if vectors.size == 0 or vectors.shape[0] < 2:
        return "Unable to compute summary due to insufficient data."
    
    # Compute similarity matrix
    similarity_matrix = cosine_similarity(vectors)
    
    # Use PageRank algorithm to rank sentences
    scores = nx.pagerank(nx.from_numpy_array(similarity_matrix))
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    
    # Generate summary from top ranked sentences
    summary = " ".join([s[1] for s in ranked_sentences[:3]])  # Adjust summary length as needed
    return summary

# Function to determine summarization type based on user input
def summarize_text(input_type, text, file, summarization_type):
    if input_type == "Text":
        text = text
    elif input_type == "PDF":
        text = extract_text_from_pdf(file.name)
    else:
        return "Invalid input type selected."

    if summarization_type == "Abstractive":
        return abstractive_summarize(text)
    elif summarization_type == "Extractive":
        return extractive_summarize(text)

# Function to extract text from a PDF file using PyMuPDF (fitz)
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)  # Open PDF file from path
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# Description content for the description page using Markdown
description_content = """
<div class='description-content'>

# WELLCOME TO TEXT SUMMIFY!! 

This interface allows you to summarize text using two different methods:

## Abstractive Summarization
This method generates a summary by interpreting the main ideas of the text and rephrasing them in a concise manner. It uses a fine-tuned T5 model to perform this task.

## Extractive Summarization
This method selects the most important sentences from the text and combines them to form a summary. It uses the TextRank algorithm for this purpose.

To use the interface, choose the input type (Text or PDF), enter your text or upload a PDF file, select the type of summarization (Abstractive or Extractive), and click the "Submit" button to get your summary.

## Features
- Supports both abstractive and extractive summarization techniques.
- Utilizes a fine-tuned T5 model for high-quality abstractive summaries.
- Implements the TextRank algorithm for effective extractive summaries.
- Easy-to-use interface with enhanced styling and graphics.

## Discover the power of TEXT SUMMIFY today !! and streamline your interaction with textual information like never before!
</div>
</div>
"""

# Define Gradio interface for summarization
def input_type_change(input_type):
    if input_type == "Text":
        return gr.update(visible=True), gr.update(visible=False)
    elif input_type == "PDF":
        return gr.update(visible=False), gr.update(visible=True)

with gr.Blocks() as summarize_interface:
    gr.HTML(open("styles.html").read())  # Include external HTML file for styles
    gr.Markdown("# <span id='text-summify-heading'>TEXT SUMMIFY</span>")
    
    with gr.Row():
        input_type = gr.Radio(["Text", "PDF"], label="Input Type", elem_id="input-type")
        summarization_type = gr.Dropdown(['Abstractive', 'Extractive'], label="Summarization Type", elem_id="summarization-type")
    
    text_input = gr.Textbox(lines=10, placeholder="Enter Text Here...", visible=True, elem_id="input-area")
    pdf_input = gr.File(label="Upload PDF File", visible=False, elem_id="input-area")
    
    input_type.change(input_type_change, input_type, [text_input, pdf_input])
    
    summarize_button = gr.Button("Submit", elem_id="btn-submit")
    clear_button = gr.Button("Clear", elem_id="btn-clear")
    output = gr.Textbox(label="Summary", elem_id="output-area")
    
    def clear_function():
        return "", None, ""

    summarize_button.click(summarize_text, [input_type, text_input, pdf_input, summarization_type], output)
    clear_button.click(clear_function, outputs=[text_input, pdf_input, output])

# Define the description interface using Markdown
description_interface = gr.Markdown(description_content)

# Combine both interfaces into a tabbed interface
combined_interface = gr.TabbedInterface(
    [summarize_interface, description_interface],
    ["Summarize Text", "About TEXT SUMMIFY"]
)

# Launch the combined interface
combined_interface.launch()
