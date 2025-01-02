import streamlit as st
import base64
import string
import nltk
import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
from langdetect import detect
from sklearn.feature_extraction.text import CountVectorizer
from docx import Document
import seaborn as sns
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import PyPDF2

# Download necessary NLTK data
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")

# Helper function: Load and encode an image to base64 for CSS
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# Helper function: Save transformed text for download
def save_transformed_text(original_text, transformed_text, filename="transformed_text.txt"):
    with open(filename, "w") as f:
        f.write("Original Text:\n")
        f.write(original_text)
        f.write("\n\nTransformed Text:\n")
        f.write(transformed_text)
    return filename

# Helper function: Read .docx file and extract text
def read_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# Function to process text from PDF
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# Image path and base64 encoding
image_path = "E:/NLP/WhatsApp Image 2024-11-15 at 02.36.15_2f944400.jpg"
image_base64 = get_base64_image(image_path)

# Streamlit app CSS for styling
st.markdown(
    f"""
    <style>
    .header {{
        text-align: center;
        font-size: 3em;
        font-weight: bold;
        background: -webkit-linear-gradient(45deg, #f3ec78, #af4261);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 20px;
    }}
    .circle {{
        position: absolute;
        top: 20px;
        right: 20px;
        width: 150px;
        height: 150px;
        border-radius: 50%;
        border: 5px solid #af4261;
        background-image: url("data:image/jpeg;base64,{image_base64}");
        background-size: cover;
        background-position: center;
    }}
    .creator {{
        text-align: center;
        font-size: 1.2em;
        font-style: italic;
        color: #333;
        margin-top: 30px;
    }}
    .scrollable-box {{
        height: 300px;
        overflow-y: scroll;
        border: 2px solid #af4261;
        padding: 10px;
        margin-bottom: 20px;
        background-color: #f9f9f9;
    }}
    .box {{
        width: 100%;
        max-width: 800px;
        margin: 20px auto;
        padding: 10px;
        border: 2px solid #af4261;
        background-color: #f9f9f9;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# App Title and Creator Info
st.markdown('<div class="circle"></div>', unsafe_allow_html=True)
st.markdown('<div class="header">BhashaShutra</div>', unsafe_allow_html=True)
st.markdown('<div class="creator">Creator: Padhi Ashish</div>', unsafe_allow_html=True)

# File upload and text input
uploaded_file = st.file_uploader("Upload a file", type=["txt", "docx", "csv", "xlsx", "pdf"])

# Initialize text variable
text = ""

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1]

    # Process different file types
    if file_type == "txt":
        text = uploaded_file.read().decode("utf-8")
    
    elif file_type == "docx":
        text = read_docx(uploaded_file)
    
    elif file_type == "pdf":
        text = extract_text_from_pdf(uploaded_file)
    
    elif file_type in ["csv", "xlsx"]:
        df = pd.read_csv(uploaded_file) if file_type == "csv" else pd.read_excel(uploaded_file)
        st.write("### File Preview")
        st.dataframe(df.head())  # Preview the first 5 rows of the file
        st.markdown("**Features coming soon for CSV/Excel files!**")
        text = ""  # No text processing for these file types
        
else:
    st.markdown("### Input Text")
    text = st.text_area("Enter your text here:", height=200)

# Initialize transformed text
transformed_text = text

# Sidebar options (all functions visible)
st.sidebar.markdown("### Basic Functions")
count_words = st.sidebar.checkbox("Count Words")
count_punctuation = st.sidebar.checkbox("Count Punctuation")
show_most_repeated_word = st.sidebar.checkbox("Show Most Repeated Word")
show_least_repeated_word = st.sidebar.checkbox("Show Least Repeated Word")
convert_lowercase = st.sidebar.checkbox("Convert to Lowercase")
convert_uppercase = st.sidebar.checkbox("Convert to Uppercase")

st.sidebar.markdown("### Advanced Functions")
remove_punctuation = st.sidebar.checkbox("Remove Punctuation")
remove_stopwords = st.sidebar.checkbox("Remove Stopwords")
sentence_tokenization = st.sidebar.checkbox("Sentence Tokenization")
word_tokenization = st.sidebar.checkbox("Word Tokenization")
perform_stemming = st.sidebar.checkbox("Perform Stemming")
perform_lemmatization = st.sidebar.checkbox("Perform Lemmatization")
pos_tagging = st.sidebar.checkbox("POS Tagging")

st.sidebar.markdown("### Text Visualization")
generate_wordcloud = st.sidebar.checkbox("Generate Word Cloud")
word_frequency_plot = st.sidebar.checkbox("Word Frequency Plot")

st.sidebar.markdown("### Sentiment Analysis")
sentiment_analysis = st.sidebar.checkbox("Analyze Sentiment")

st.sidebar.markdown("### Language Detection")
language_detection = st.sidebar.checkbox("Detect Language")

st.sidebar.markdown("### N-Gram Analysis")
ngram_analysis = st.sidebar.checkbox("Analyze N-Grams")
n = st.sidebar.slider("Select N for N-Grams", 1, 5, 2)  # Slider for selecting N in N-grams

# Process text transformations based on selected checkboxes
if text:
    if count_words:
        word_count = len(word_tokenize(transformed_text))
        st.write(f"**Total Words:** {word_count}")
    if count_punctuation:
        punctuation_count = sum(1 for char in transformed_text if char in string.punctuation)
        st.write(f"**Total Punctuation Marks:** {punctuation_count}")
    if show_most_repeated_word:
        words = word_tokenize(transformed_text)
        most_common = Counter(words).most_common(1)
        if most_common:
            st.write(f"**Most Repeated Word:** {most_common[0][0]} ({most_common[0][1]} times)")
    if show_least_repeated_word:
        words = word_tokenize(transformed_text)
        least_common = [word for word, count in Counter(words).items() if count == 1]
        if least_common:
            st.write(f"**Least Repeated Word:** {least_common[0]}")
    if convert_lowercase:
        transformed_text = transformed_text.lower()
    if convert_uppercase:
        transformed_text = transformed_text.upper()

    if remove_punctuation:
        transformed_text = ''.join(char for char in transformed_text if char not in string.punctuation)
    if remove_stopwords:
        stop_words = set(stopwords.words("english"))
        transformed_text = " ".join(word for word in word_tokenize(transformed_text) if word.lower() not in stop_words)
    if sentence_tokenization:
        sentences = sent_tokenize(transformed_text)
        transformed_text = "\n".join(sentences)
    if word_tokenization:
        words = word_tokenize(transformed_text)
        transformed_text = " ".join(words)
    if perform_stemming:
        stemmer = PorterStemmer()
        transformed_text = " ".join(stemmer.stem(word) for word in word_tokenize(transformed_text))
    if perform_lemmatization:
        lemmatizer = WordNetLemmatizer()
        transformed_text = " ".join(lemmatizer.lemmatize(word) for word in word_tokenize(transformed_text))
    if pos_tagging:
        pos_tags = nltk.pos_tag(word_tokenize(transformed_text))
        transformed_text = str(pos_tags)
    if generate_wordcloud:  
        wordcloud = WordCloud(width=800, height=400).generate(transformed_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)
    if word_frequency_plot:
        vectorizer = CountVectorizer(stop_words="english")
        word_counts = vectorizer.fit_transform([transformed_text])
        freq = sum(word_counts.toarray())
        word_freq = pd.DataFrame(list(zip(vectorizer.get_feature_names_out(), freq)), columns=["Word", "Frequency"])
        st.write("**Word Frequency Plot:**")
        sns.barplot(x="Frequency", y="Word", data=word_freq.nlargest(10, "Frequency"))
        st.pyplot(plt)
    if sentiment_analysis:
        sentiment = TextBlob(transformed_text).sentiment
        st.write(f"**Sentiment Analysis:** {sentiment}")
    if language_detection:
        try:
            language = detect(transformed_text)
            st.write(f"**Detected Language:** {language}")
        except:
            st.write("Error detecting language.")
    if ngram_analysis:
        ngrams = nltk.ngrams(word_tokenize(transformed_text), n)
        ngram_freq = Counter(ngrams)
        st.write(f"**Most Common {n}-Grams:**")
        st.write(ngram_freq.most_common(10))

# Display original and transformed text in scrollable boxes with unique keys
st.markdown("### Original and Transformed Text")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Original Text**", unsafe_allow_html=True)
    st.text_area("Original Text", text, height=300, key="original_text")

with col2:
    st.markdown("**Transformed Text**", unsafe_allow_html=True)
    st.text_area("Transformed Text", transformed_text, height=300, key="transformed_text")

# Option to download transformed text
if text:
    download_button = save_transformed_text(text, transformed_text)
    st.download_button("Download Transformed Text", download_button)
