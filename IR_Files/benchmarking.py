import json
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

dataset = "trec-covid"  # Change this to the actual dataset being used
corpus, queries, qrels = GenericDataLoader(dataset + "/").load(split="test")

results_file = 'Results.json'
with open(results_file, 'r') as file:
    raw_results = json.load(file)

evaluator = EvaluateRetrieval()

beir_results = {}
for query_id, docs in raw_results.items():
    beir_results[query_id] = {doc_id: score for doc_id, score in docs}

results = evaluator.evaluate(qrels, beir_results, k_values=[1, 5, 10, 100])

print("Evaluation Results:", results)
