import xml.etree.ElementTree as ET
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
from wordcloud import WordCloud
import streamlit as st
import requests

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Function to extract text from XML
def extract_text_from_xml(xml_content):
    tree = ET.ElementTree(ET.fromstring(xml_content))
    root = tree.getroot()
    text = ' '.join([elem.text for elem in root.iter() if elem.text])
    return text

# Function to process text
def process_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token.isalpha() and token.lower() not in stop_words]
    sentences = sent_tokenize(text)
    return tokens, sentences

# Function to compute frequencies
def compute_frequencies(tokens, sentences):
    token_freq = Counter(tokens)
    sentence_freq = Counter(sentences)
    total_tokens = sum(token_freq.values())
    weighted_token_freq = {token: freq / total_tokens for token, freq in token_freq.items()}
    return token_freq, weighted_token_freq, sentence_freq

# Function to generate word cloud
def generate_word_cloud(tokens):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(tokens))
    return wordcloud

# Function to summarize text
def summarize_text(text, ratio=0.2):
    sentences = sent_tokenize(text)
    num_sentences = max(1, int(len(sentences) * ratio))
    return ' '.join(sentences[:num_sentences])

# Streamlit application
def main():
    st.title("AI Application for XML Text Processing")
    
    input_type = st.radio("Select input type:", ("XML File", "XML URL"))

    text = None
    
    if input_type == "XML File":
        uploaded_file = st.file_uploader("Choose an XML file", type="xml")
        if uploaded_file is not None:
            xml_content = uploaded_file.read().decode('utf-8')
            text = extract_text_from_xml(xml_content)
    else:
        xml_url = st.text_input("Enter XML URL")
        if xml_url:
            response = requests.get(xml_url)
            if response.status_code == 200:
                xml_content = response.text
                text = extract_text_from_xml(xml_content)
            else:
                st.error("Failed to retrieve XML from the URL")
    
    if text:
        st.write("Extracted Text:")
        st.write(text)
        
        tokens, sentences = process_text(text)
        token_freq, weighted_token_freq, sentence_freq = compute_frequencies(tokens, sentences)
        
        st.write("Token Frequency:")
        st.write(token_freq)
        
        st.write("Weighted Token Frequency:")
        st.write(weighted_token_freq)
        
        st.write("Sentence Frequency:")
        st.write(sentence_freq)
        
        wordcloud = generate_word_cloud(tokens)
        st.write("Word Cloud:")
        st.image(wordcloud.to_array())
        
        summary_ratio = st.slider("Select summarization percentage", min_value=10, max_value=100, value=20)
        summary_ratio /= 100  # Convert percentage to a ratio
        
        summary = summarize_text(text, ratio=summary_ratio)
        st.write("Text Summary:")
        st.write(summary)

if __name__ == "__main__":
    main()
