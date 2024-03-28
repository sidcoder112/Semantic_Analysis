# Import necessary libraries
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# NLTK setup (download required resources)
nltk.download('punkt')
nltk.download('stopwords')

# Load training data without header
training_data = pd.read_csv('twitter_training.csv', header=None)
training_data.columns = ['Tweet ID', 'Entity', 'Sentiment', 'Tweet Content']

# Load validation data without header
validation_data = pd.read_csv('twitter_validation.csv', header=None)
validation_data.columns = ['Tweet ID', 'Entity', 'Sentiment', 'Tweet Content']


# Function to clean text
def clean_text(text):
    # Check if the value is NaN or null
    if pd.isna(text) or pd.isnull(text):
        return ''  # Return an empty string if NaN or null
    # Remove special characters and URLs
    text = re.sub(r'\W+', ' ', str(text))  # Convert to string before applying regex
    text = re.sub(r'http\S+', '', text)

    # Convert text to lowercase
    text = text.lower()

    return text


# Tokenization
def tokenize_text(text):
    return word_tokenize(text)


# Remove stopwords
stop_words = set(stopwords.words('english'))


def remove_stopwords(tokens):
    return [token for token in tokens if token not in stop_words]


# Stemming
stemmer = PorterStemmer()


def stem_tokens(tokens):
    return [stemmer.stem(token) for token in tokens]


# Apply preprocessing steps
def preprocess_data(data):
    # Clean text and handle missing values
    data['cleaned_text'] = data['Tweet Content'].apply(clean_text)
    # Tokenization
    data['tokens'] = data['cleaned_text'].apply(tokenize_text)
    # Remove stopwords
    data['filtered_tokens'] = data['tokens'].apply(remove_stopwords)
    # Stemming
    data['stemmed_tokens'] = data['filtered_tokens'].apply(stem_tokens)
    return data


# Preprocess training data
training_data = preprocess_data(training_data)

# Preprocess validation data
validation_data = preprocess_data(validation_data)

# Display preprocessed data (optional)
print("Preprocessed Training Data:")
print(training_data.head())
print("\nPreprocessed Validation Data:")
print(validation_data.head())
