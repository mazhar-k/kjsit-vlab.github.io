import re
from transformers import pipeline
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import networkx as nx
import nltk
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')

def count_sentences(text):
    sentences = nltk.sent_tokenize(text)
    return len(sentences)


def rule_based_summarization(text):
    num_sentences = count_sentences(text)
    # Split the text into individual sentences
    sentences = re.split('(?<=[.!?]) +', text)
    
    # Calculate the importance score of each sentence based on predefined rules
    scores = []
    for sentence in sentences:
        score = 0
        # Rule 1: Add 2 points for each uppercase word
        score += len(re.findall(r'\b[A-Z][A-Z]+\b', sentence)) * 2
        # Rule 2: Add 1 point for each word that appears in a predefined list of important words
        important_words = ['important', 'key', 'essential']
        score += sum([1 for word in sentence.split() if word.lower() in important_words])
        # Rule 3: Add 1 point for each word that appears in a predefined list of topic words
        topic_words = ['topic', 'subject', 'theme']
        score += sum([1 for word in sentence.split() if word.lower() in topic_words])
        scores.append(score)
    
    # Get the indices of the top num_sentences sentences based on importance score
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:num_sentences]
    
    # Combine the top sentences into a summary
    summary = ' '.join([sentences[i] for i in top_indices])
    return summary

def abstractive_summarization(text):
    max_length = len(text) // 2
    # Load the T5 model and tokenizer
    model = pipeline('text2text-generation', model='t5-base', tokenizer='t5-base', device=0)
    
    # Generate a summary using the T5 model
    summary = model(text, max_length=max_length, do_sample=True, num_beams=4)[0]['generated_text']
    return summary

def clean_text_fun(text):
    # Remove non-alphanumeric characters and extra white space
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def build_similarity_matrix(sentences):
    # Vectorize the sentences using TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    
    # Compute the pairwise cosine similarity between sentences
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix

def graph_based_summarization(text):
    num_sentences = count_sentences(text) // 2
    # Clean the text and tokenize into sentences
    clean_text = clean_text_fun(text)
    sentences = sent_tokenize(clean_text)
    
    # Build the similarity matrix and compute the PageRank scores
    similarity_matrix = build_similarity_matrix(sentences)
    pagerank_scores = np.squeeze(np.asarray(np.sum(similarity_matrix, axis=1)))
    
    # Get the indices of the top PageRank scores
    top_indices = pagerank_scores.argsort()[-num_sentences:][::-1]
    
    # Sort the indices and get the corresponding sentences
    top_sentences = [sentences[i] for i in sorted(top_indices)]
    
    # Join the sentences and return the summary
    summary = ' '.join(top_sentences)
    return summary


def extractive_summarization(text):
    num_sentences = count_sentences(text)
    # Load the English language model in spaCy
    nlp = spacy.load('en_core_web_sm')
    
    # Tokenize the text and remove stop words
    doc = nlp(text)
    tokens = [token for token in doc if not token.is_stop]
    
    # Calculate the frequency of each word
    word_freq = Counter([token.text for token in tokens])
    
    # Calculate the score of each sentence based on the frequency of its words
    sentence_scores = {}
    for sent in doc.sents:
        for token in sent:
            if token.text in word_freq.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_freq[token.text]
                else:
                    sentence_scores[sent] += word_freq[token.text]
    
    # Get the top num_sentences sentences based on their scores
    top_sents = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
    
    # Combine the top sentences into a summary
    summary = ' '.join([sent.text for sent, score in top_sents])
    return summary

if __name__ == "__main__":
    text = 'Machine learning is an important component of the growing field of data science. Through the use of statistical methods, algorithms are trained to make classifications or predictions, and to uncover key insights in data mining projects. These insights subsequently drive decision making within applications and businesses, ideally impacting key growth metrics. As big data continues to expand and grow, the market demand for data scientists will increase. They will be required to help identify the most relevant business questions and the data to answer them.'
    # print(rule_based_summarization(text))
    print(abstractive_summarization(text))
    # print(extractive_summarization(text))
    # print(graph_based_summarization(text))