import json

def load_results(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def combine_scores(scores1, scores2, weight1=0.5, weight2=0.5):
    combined_scores = {}
    for doc_id, score in scores1:
        combined_scores[doc_id] = score * weight1
    for doc_id, score in scores2:
        if doc_id in combined_scores:
            combined_scores[doc_id] += score * weight2
        else:
            combined_scores[doc_id] = score * weight2
    # Convert combined_scores back to a sorted list of tuples
    combined_scores = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
    return combined_scores

def combine_results(file1, file2, weight1=0.5, weight2=0.5, output_file='Results.json'):
    print(f"Loading {file1}")
    results1 = load_results(file1)
    print(f"Loading {file2}")
    results2 = load_results(file2)

    combined_results = {}
    print("Combining results")
    for query_id in results1:
        if query_id in results2:
            combined_results[query_id] = combine_scores(results1[query_id], results2[query_id], weight1, weight2)
        else:
            combined_results[query_id] = results1[query_id]

    for query_id in results2:
        if query_id not in combined_results:
            combined_results[query_id] = results2[query_id]

    # Save the combined results to a new file
    with open(output_file, 'w') as file:
        json.dump(combined_results, file, indent=4)
    print(f"Combined results saved to {output_file}")

# Replace the file names with actual models used
file1 = 'Results (msmarco-roberta-base-ance-firstp).json'
file2 = 'Results (BM25).json'
combine_results(file1, file2, weight1=0.5, weight2=0.5)
