import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from utils import progress_bar
import time
import json

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def tokenize(text):
    """
    Tokenize the text using NLTK's word_tokenize
    """
    tokens = word_tokenize(text.lower())
    return tokens

def stem_tokens(tokens):
    return [stemmer.stem(token) for token in tokens]

def remove_extras(tokens):
    return [token for token in tokens if token not in ['no_queri', 'no_narr']]

def preprocess_text(text):
    tokens = tokenize(text)
    tokens = stem_tokens(tokens)
    tokens = remove_extras(tokens)
    return tokens

def preprocess_documents(documents):
    previousId = "t"
    count = 1
    start_time = time.time()

    for doc in documents:
        fileId = str(doc['DOCNO'].split(" ")[0])
        if (not (fileId == previousId)):
            #print("Doc File: " +str(count))
            progress_bar(count, len(documents))
            previousId = fileId
            count = count + 1
        doc['TEXT'] = preprocess_text(doc['TEXT'])
        doc['HEAD'] = preprocess_text(doc['HEAD'])
    
    end_time = time.time()
    print(f"\nTime taken to parse and preprocess documents: {end_time - start_time:.2f} seconds")
    return documents

def preprocess_queries(queries):
    for query in queries:
        query['title'] = preprocess_text(query['title'])
        query['query'] = preprocess_text(query['query'])
        query['narrative'] = preprocess_text(query['narrative'])
    return queries

def save_preprocessed_data(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

def load_preprocessed_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data