import json

def build_inverted_index(documents):
    """
    Build an inverted index from the preprocessed documents
    """
    inverted_index = {}
    for doc in documents:
        doc_id = doc['DOCNO']
        text_tokens = doc['TEXT']
        for token in text_tokens:
            if token not in inverted_index:
                inverted_index[token] = {}
            if doc_id not in inverted_index[token]:
                inverted_index[token][doc_id] = 0
            inverted_index[token][doc_id] += 1
    return inverted_index

def calculate_document_lengths(documents):
    """
    Calculate the length of each document based on the number of terms
    """
    doc_lengths = {}
    for doc in documents:
        doc_id = doc['DOCNO']
        text_tokens = doc['TEXT']
        doc_length = len(text_tokens)
        doc_lengths[doc_id] = doc_length
    return doc_lengths

def save_inverted_index(inverted_index, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(inverted_index, file, indent=4)

def load_inverted_index(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        inverted_index = json.load(file)
    return inverted_index
