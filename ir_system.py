import json
import re
import math
import sys
import collections
from os import path

CORPUS_FILE = 'scifact/scifact/corpus.jsonl'
QUERIES_FILE = 'scifact/scifact/queries.jsonl'
STOPWORDS_FILE = 'List of Stopwords.html'
OUTPUT_FILE = 'Results.txt'
RUN_TAG = 'BM25_Run_1'

# BM25 Parameters (Industry Standard)
k1 = 1.2
b = 0.75

#PREPROCESSING
def load_stopwords(filepath):
    """Parses the HTML stopwords file."""
    stopwords = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            # Extract text inside <pre> tags or just raw text if no tags found
            if '<pre>' in content:
                content = content.split('<pre>')[1].split('</pre>')[0]
            
            for line in content.splitlines():
                word = line.strip().lower()
                if word:
                    stopwords.add(word)
    except FileNotFoundError:
        print(f"Warning: {filepath} not found. Proceeding without stopwords.")
    return stopwords

STOPWORDS = load_stopwords(STOPWORDS_FILE)

def preprocess(text, stemming=True):
    if not text:
        return []
    
    # 1. Lowercase and remove non-alphanumeric chars (keep spaces)
    text = text.lower()
    # Replace non-alphanumeric with space
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # 2. Tokenize by splitting on whitespace
    tokens = text.split()

# 3. Remove Stopwords and Short words
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    
    # 4. Porter Stemming (if NLTK is available)
    try:
        from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]
    except ImportError:
        pass
        
    return tokens

#INDEXING
def build_index(corpus_path, use_full_text=True):
    print(f"Indexing corpus from {corpus_path}...")
    
    inverted_index = collections.defaultdict(list)
    doc_lengths = {}
    
    avg_doc_length = 0
    total_tokens = 0
    num_docs = 0
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            doc_id = data['_id']
            
            # Combine title and text
            content = data['title']
            if use_full_text:
                content += " " + data['text']
            
            tokens = preprocess(content)
            length = len(tokens)
            
            doc_lengths[doc_id] = length
            total_tokens += length
            num_docs += 1
            
            term_counts = collections.Counter(tokens)
            
            for term, tf in term_counts.items():
                inverted_index[term].append((doc_id, tf))
            
            # Show the progress on the processing
            if num_docs % 2000 == 0:
                print(f"Processed {num_docs} documents...")

    if num_docs > 0:
        avg_doc_length = total_tokens / num_docs
    
    # Print number of unique terms and average document length for report
    print(f"Indexing complete. {len(inverted_index)} unique terms. Avg doc len: {avg_doc_length:.2f}")
    return inverted_index, doc_lengths, avg_doc_length, num_docs

#RETRIEVAL & RANKING 
def score_query(query_tokens, inverted_index, doc_lengths, avg_doc_length, num_docs):
    #Creates a dictionary to store scores for each document (with missing keys automatically starting at 0.0)
    scores = collections.defaultdict(float)
    
    #Processes each word in the preprocessed query
    #If a term doesn't exist in any document, skip it (no matches possible)
    for term in query_tokens:
        if term not in inverted_index:
            continue
            
        
        #Retrieves the list of documents containing this term in this format: [(doc_id1, tf1), (doc_id2, tf2), ...]
        postings = inverted_index[term]
        
        #Calculate IDF
        n_docs_with_term = len(postings)
        idf = math.log((num_docs - n_docs_with_term + 0.5) / (n_docs_with_term + 0.5) + 1)
        

        #Compute score for each document containing the term
        for doc_id, tf in postings:
            doc_len = doc_lengths[doc_id]
            
            # BM25 Formula = IDF * (TF * (k1 + 1)) / (TF + k1 * (1 - b + b * (doc_len / avgdl)))
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / avg_doc_length))
            
            scores[doc_id] += idf * (numerator / denominator)
            
    return scores

#Orchestrates the entire system
def run_system(use_full_text=True, output_file=None, run_tag=None):
    if output_file is None:
        output_file = OUTPUT_FILE
    if run_tag is None:
        run_tag = RUN_TAG
        
    # 1. Build Index
    index, doc_lens, avg_len, N = build_index(CORPUS_FILE, use_full_text)
    
    # 2. Process Queries
    print("Processing queries...")
    results = []
    
    with open(QUERIES_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            q_data = json.loads(line)
            q_id = q_data['_id']
            q_text = q_data['text']
            
            # filter only odd numbered queries
            if int(q_id) % 2 == 0:
                continue
            
            q_tokens = preprocess(q_text)
            
            # Get Scores
            doc_scores = score_query(q_tokens, index, doc_lens, avg_len, N)
            
            # Sort and Rank (Top 100)
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:100]
            
            for rank, (doc_id, score) in enumerate(sorted_docs, 1):
                results.append((q_id, doc_id, rank, score))
    
    # 3. Write Results
    print(f"Writing results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        # Sort results by Query ID then Rank
        results.sort(key=lambda x: (int(x[0]), x[2]))
        
        for q_id, doc_id, rank, score, in results:
            f.write(f"{q_id} Q0 {doc_id} {rank} {score:.4f} {run_tag}\n")

    # Debug/Stats for Report
    print("\n--- Statistics for Report ---")
    print(f"Vocabulary Size: {len(index)}")
    print("Sample 100 tokens:", list(index.keys())[:100])
    
    print("\n--- First 10 Answers for First 2 Queries ---")
    unique_queries = sorted(list(set(r[0] for r in results)), key=lambda x: int(x))
    for q_id in unique_queries[:2]:
        print(f"Query {q_id}:")
        q_res = [r for r in results if r[0] == q_id][:10]
        for item in q_res:
            print(f"  Rank {item[2]}: Doc {item[1]} (Score: {item[3]:.4f})")

if __name__ == "__main__":
    try:
        import nltk
    except ImportError:
        print("NLTK not found. Install for better performance: pip install nltk")
    
    print("="*60)
    print("EXPERIMENT 1: Title-Only")
    print("="*60)
    run_system(use_full_text=False, output_file='Results_TitleOnly.txt', run_tag='BM25_TitleOnly_Run')
    
    print("\n" + "="*60)
    print("EXPERIMENT 2: Title + Full Text")
    print("="*60)
    run_system(use_full_text=True, output_file='Results_FullText.txt', run_tag='BM25_FullText_Run')