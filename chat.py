import requests
from bs4 import BeautifulSoup
import re
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from transformers import pipeline

# Download NLTK data
nltk.data.path.append("./nltk_data")
nltk.download("punkt")
nltk.download("stopwords")


# Web Scraping
def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    text = ' '.join([p.get_text() for p in soup.find_all('p')])
    return text


url = "https://en.wikipedia.org/wiki/Machine_learning"
raw_text = scrape_website(url)


# Text Cleaning and Preprocessing
def clean_text(text):
    text = re.sub(r'\[.*?\]', '', text)  # Remove references [1], [2], etc.
    text = re.sub(r'[^a-zA-Z\s.]', '', text)  # Remove special characters but keep sentences
    text = text.lower()
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words("english"))
    cleaned_sentences = [' '.join([word for word in word_tokenize(sent) if word not in stop_words]) for sent in
                         sentences]
    return sentences, cleaned_sentences


original_sentences, cleaned_sentences = clean_text(raw_text)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cleaned_sentences)

# NLP-based Q&A using Transformers
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")


def answer_question(question):
    try:
        response = qa_pipeline({"question": question, "context": raw_text})
        return response.get("answer", "I couldn't find an answer.")
    except Exception as e:
        return f"Error processing your question: {e}"


# Search Website Content using TF-IDF and Cosine Similarity
def search_website(query):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, X).flatten()
    best_idx = np.argmax(similarities)
    if similarities[best_idx] > 0.1:  # Ensure relevance threshold
        return original_sentences[best_idx]
    return None


def chatbot():
    print("Ask me anything about the scraped website!")
    while True:
        query = input("You: ").strip().lower()
        if query in ["exit", "quit", "stop"]:
            print("Goodbye!")
            break
        if not query:
            print("Bot: Please enter a valid question.")
            continue

        try:
            website_response = search_website(query)
            if website_response:
                print("Bot:", website_response)
            else:
                response = answer_question(query)
                print("Bot:", response)
        except Exception as e:
            print(f"Bot: Sorry, I couldn't process that question. Error: {e}")


if __name__ == "__main__":
    chatbot()