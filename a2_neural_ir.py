import json
import collections
from sentence_transformers import SentenceTransformer, CrossEncoder, util

CORPUS_FILE = 'scifact/scifact/corpus.jsonl'
QUERIES_FILE = 'scifact/scifact/queries.jsonl'
A1_RESULTS = 'Results_FullText.txt'

def get_candidates():
    """Loads top 100 candidate doc IDs per query from Assignment 1."""
    candidates = collections.defaultdict(list)
    with open(A1_RESULTS, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 6:
                candidates[parts[0]].append(parts[2])
    return candidates

def load_data(candidates):
    """Loads only the needed queries and candidate document texts to save memory."""
    queries = {}
    with open(QUERIES_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if int(data['_id']) % 2 != 0: # Odd queries only
                queries[data['_id']] = data['text']

    # Flatten candidate list to know which docs to load
    needed_docs = set(doc_id for docs in candidates.values() for doc_id in docs)
    corpus = {}
    
    with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if data['_id'] in needed_docs:
                corpus[data['_id']] = data['title'] + " " + data['text']
                
    return queries, corpus

def write_trec(results_dict, filename, run_tag):
    """Writes results to TREC format."""
    with open(filename, 'w', encoding='utf-8') as f:
        # Sort queries numerically
        for q_id in sorted(results_dict.keys(), key=int):
            for rank, (doc_id, score) in enumerate(results_dict[q_id], 1):
                f.write(f"{q_id} Q0 {doc_id} {rank} {score:.4f} {run_tag}\n")

def print_top_10(results_dict, method_name):
    """Prints top 10 results for queries 1 and 3 for the README."""
    print(f"\n--- Top 10 for {method_name} ---")
    for q_id in ['1', '3']:
        if q_id in results_dict:
            print(f"Query {q_id}:")
            for rank, (doc_id, score) in enumerate(results_dict[q_id][:10], 1):
                print(f"  Rank {rank}: Doc {doc_id} (Score: {score:.4f})")

def main():
    print("Loading data...")
    candidates = get_candidates()
    queries, corpus = load_data(candidates)
    
    # METHOD 1: Bi-Encoder (Sentence-BERT)
    print("Running Method 1: Bi-Encoder...")
    bi_model = SentenceTransformer('all-MiniLM-L6-v2')
    bi_results = collections.defaultdict(list)
    
    for q_id, q_text in queries.items():
        if q_id not in candidates: continue
        doc_ids = candidates[q_id]
        doc_texts = [corpus[d] for d in doc_ids]
        
        q_emb = bi_model.encode(q_text, convert_to_tensor=True)
        doc_embs = bi_model.encode(doc_texts, convert_to_tensor=True)
        scores = util.cos_sim(q_emb, doc_embs)[0].cpu().tolist()
        
        ranked = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)
        bi_results[q_id] = ranked

    print_top_10(bi_results, "Bi-Encoder")
    write_trec(bi_results, 'Results_BiEncoder.txt', 'BiEncoder')

    # METHOD 2: Cross-Encoder
    print("\nRunning Method 2: Cross-Encoder...")
    cross_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    cross_results = collections.defaultdict(list)
    
    for q_id, q_text in queries.items():
        if q_id not in candidates: continue
        doc_ids = candidates[q_id]
        pairs = [[q_text, corpus[d]] for d in doc_ids]
        
        scores = cross_model.predict(pairs)
        ranked = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)
        cross_results[q_id] = ranked

    print_top_10(cross_results, "Cross-Encoder")
    # Named simply "Results" as requested for the best system
    write_trec(cross_results, 'Results.txt', 'CrossEncoder')
    
    print("\nDone! Run trec_eval.py on Results_BiEncoder.txt and Results.txt to get your MAP and P@10 scores.")

if __name__ == "__main__":
    main()