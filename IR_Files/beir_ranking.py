# beir_ranking.py
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.search.sparse import SparseSearch
from beir.reranking.models import CrossEncoder
from beir.reranking import Rerank
from beir.retrieval.evaluation import EvaluateRetrieval
import tensorflow as tf
import json
from ranking import BM25

def load_model(model_name, model_type, documents=None, inverted_index=None, doc_lengths=None):
    if model_type == "bm25":
        if documents is None or inverted_index is None or doc_lengths is None:
            raise ValueError("Documents, inverted_index, and doc_lengths are required for BM25.")
        return BM25(inverted_index, doc_lengths)
    elif model_type == "sparta":
        return SparseSearch(models.SPARTA(model_name), batch_size=128)
    elif model_type in ["sentence-bert", "ance"]:
        return DRES(models.SentenceBERT(model_name), batch_size=16)
    elif model_type == "use-qa":
        return DRES(models.UseQA(model_name))
    elif model_type == "dpr":
        return DRES(models.SentenceBERT((
    "facebook-dpr-question_encoder-multiset-base",
    "facebook-dpr-ctx_encoder-multiset-base",
    " [SEP] "), batch_size=128))
    elif model_type == "cross-encoder":
        return CrossEncoder(model_name)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
def combine_scores(scores1, scores2, weight1=0.5, weight2=0.5):
    combined_scores = {}
    for doc_id in scores1:
        combined_scores[doc_id] = scores1[doc_id] * weight1
    for doc_id in scores2:
        if doc_id in combined_scores:
            combined_scores[doc_id] += scores2[doc_id] * weight2
        else:
            combined_scores[doc_id] = scores2[doc_id] * weight2
    return combined_scores

def rank_documents(documents, queries, model_name="BM25", model_type="BM25", rerank=False, inverted_index=None, doc_lengths=None):
    model = load_model(model_name, model_type, documents, inverted_index, doc_lengths)
    corpus = {}
    for doc in documents:
        corpus[doc['DOCNO']] = {
            "title": " ".join(doc['HEAD']),
            "text": " ".join(doc['TEXT'])
        }
    score_function = "cos_sim" if model_type != "bm25" else None
    if model_type == "ance": score_function = "dot"
    retriever = EvaluateRetrieval(model, score_function=score_function)
    
    # Convert queries to the correct format
    query_dict = {query['num']: " ".join(query['title'] + query['query'] + query['narrative']) for query in queries}
    
    if model_type == "bm25":
        results = model.search(corpus, query_dict)
    else:
        results = retriever.retrieve(corpus, query_dict)
    
    if rerank:
        reranker = Rerank(load_model("cross-encoder/ms-marco-electra-base", "cross-encoder"), batch_size=128)
        results = reranker.rerank(corpus, query_dict, results, top_k=100)
    
    return results

def save_results(results, output_file):
    beir_results = {}
    for query_id, docs in results.items():
        beir_results[query_id] = [(doc_id, float(score)) for doc_id, score in docs.items()]
    
    with open(output_file, 'w') as file:
        json.dump(beir_results, file, indent=4)

# Example usage:
# corpus, queries, qrels = GenericDataLoader("scifact/").load(split="test")
# results = rank_documents(corpus, queries, model_name="msmarco-distilbert-base-tas-b", model_type="sentence-bert")
# save_results(results, "Results.json")
