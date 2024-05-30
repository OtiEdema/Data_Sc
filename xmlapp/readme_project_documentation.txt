# AI XML Text Processing Application

## Overview

This project is a robust AI-powered application designed to process and analyze XML data efficiently. It leverages advanced natural language processing (NLP) techniques to extract meaningful text, perform tokenization and sentence segmentation, compute various frequency metrics, generate word clouds, and summarize text. The application is built with a user-friendly frontend using Streamlit, making it easy to upload XML files or URLs and view the processed results.

## Features

- **Text Extraction from XML**: Parses XML files or URLs to extract meaningful text content.
- **Tokenization & Sentence Segmentation**: Breaks down documents into tokens and sentences for detailed analysis.
- **Frequency Analysis**: Computes token frequency, weighted token frequency, and sentence frequency.
- **Word Cloud Generation**: Creates a visual representation of key terms using a word cloud, excluding stop words.
- **Text Summarization**: Provides customizable text summarization based on user-defined percentages.
- **Streamlit Frontend**: Interactive interface for uploading XML files or URLs and viewing the cleaned document and analysis.

## Technologies & Libraries Used

- **XML Parsing**: `xml.etree.ElementTree`
- **Natural Language Processing**: `nltk`
- **Data Visualization**: `wordcloud`
- **Web Framework**: `streamlit`
- **HTTP Requests**: `requests`
- **Data Handling**: `collections`

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/otiedema/ai-xml-text-processing.git
   cd ai-xml-text-processing

2. Create and Activate a Virtual Environment:
   python -m venv venv
   source venv/bin/activate  
   # On Windows, use `venv\Scripts\activate`

3. Install Dependencies:
   pip install -r requirements.txt
   
4. Download NLTK Data:
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')

Usage
  1. Run the Streamlit Application:
  streamlit run app.py

2. Interact with the Application:
   
   Choose between uploading an XML file or entering an XML URL.
   View the extracted text, token frequencies, weighted token frequencies, sentence frequencies, word cloud, and    
   text summary.
   Use the slider to adjust the percentage of text summarisation.

Contributing
   Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

License
   This project is licensed under the MIT License.

Contact
   For any inquiries, please reach out via LinkedIn (https://www.linkedin.com/in/oti-e-34838485/).


