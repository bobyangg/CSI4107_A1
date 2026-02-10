import sys
from collections import defaultdict

def load_qrels(qrels_file):
    # Load relevance judgments from TSV file.
    qrels = defaultdict(set)
    with open(qrels_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if line_num == 0 and line.startswith('query-id'):
                continue  
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 3:
                query_id = parts[0].strip()
                doc_id = parts[1].strip()
                relevance = int(parts[2].strip())
                if relevance > 0:
                    qrels[query_id].add(doc_id)
    return qrels

def load_results(results_file):
    # Load system results from TREC format file.
    results = defaultdict(list)
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 6:
                query_id = parts[0]
                doc_id = parts[2]
                rank = int(parts[3])
                score = float(parts[4])
                results[query_id].append((doc_id, rank, score))
    
    # Sort by rank to ensure proper order
    for query_id in results:
        results[query_id].sort(key=lambda x: x[1]) 
    
    return results

def calculate_average_precision(relevant_docs, retrieved_docs):
    # Calculate Average Precision for a single query.
    if not relevant_docs:
        return 0.0
    
    relevant_retrieved = 0
    precision_sum = 0.0
    
    for i, (doc_id, rank, score) in enumerate(retrieved_docs):
        if doc_id in relevant_docs:
            relevant_retrieved += 1
            precision_at_i = relevant_retrieved / (i + 1)
            precision_sum += precision_at_i
    
    if relevant_retrieved == 0:
        return 0.0
    
    return precision_sum / len(relevant_docs)

def calculate_map(qrels, results):
    # Calculate Mean Average Precision across all queries.
    total_ap = 0.0
    num_queries = 0
    
    query_aps = {}
    
    for query_id in qrels:
        if query_id in results:
            relevant_docs = qrels[query_id]
            retrieved_docs = results[query_id]
            ap = calculate_average_precision(relevant_docs, retrieved_docs)
            query_aps[query_id] = ap
            total_ap += ap
            num_queries += 1
            print(f"Query {query_id}: AP = {ap:.4f} (relevant: {len(relevant_docs)}, retrieved: {len(retrieved_docs)})")
    
    if num_queries == 0:
        return 0.0, query_aps
    
    map_score = total_ap / num_queries
    return map_score, query_aps

def main():
    if len(sys.argv) != 3:
        print("Usage: python evaluate_map.py <qrels_file> <results_file>")
        print("Example: python evaluate_map.py scifact/scifact/qrels/test.tsv Results_FullText.txt")
        sys.exit(1)
    
    qrels_file = sys.argv[1]
    results_file = sys.argv[2]
    
    print(f"Loading relevance judgments from: {qrels_file}")
    qrels = load_qrels(qrels_file)
    print(f"Loaded {len(qrels)} queries with relevance judgments")
    
    print(f"\nLoading results from: {results_file}")
    results = load_results(results_file)
    print(f"Loaded results for {len(results)} queries")
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    map_score, query_aps = calculate_map(qrels, results)
    
    print(f"\nMean Average Precision (MAP): {map_score:.4f}")
    print(f"Number of evaluated queries: {len(query_aps)}")
    
    # Additional statistics
    if query_aps:
        ap_values = list(query_aps.values())
        print(f"Best AP: {max(ap_values):.4f}")
        print(f"Worst AP: {min(ap_values):.4f}")
        print(f"Median AP: {sorted(ap_values)[len(ap_values)//2]:.4f}")

if __name__ == "__main__":
    main()
