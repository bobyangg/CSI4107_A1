import math
import datetime
import json

class BM25:
    def __init__(self, inverted_index, doc_lengths, k1=1.5, b=0.75, avgdl=None):
        self.inverted_index = inverted_index
        self.doc_lengths = doc_lengths
        self.k1 = k1
        self.b = b
        self.avgdl = avgdl if avgdl is not None else sum(doc_lengths.values()) / len(doc_lengths)
        self.N = len(doc_lengths)  # Total number of documents

    def idf(self, term):
        df = len(self.inverted_index.get(term, {}))
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def bm25_score(self, doc_id, query_terms):
        """
        Calculate the BM25 score for a single document and a given query
        """
        score = 0.0
        doc_length = self.doc_lengths[doc_id]
        for term in query_terms:
            if term in self.inverted_index and doc_id in self.inverted_index[term]:
                tf = self.inverted_index[term][doc_id]
                idf_value = self.idf(term)
                term_score = idf_value * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_length / self.avgdl))
                score += term_score
        return score
    
    def search(self, corpus, queries, top_k=1000):
        """
        Search method compatible with BEIR framework
        """
        results = {}
        for query_id, query in queries.items():
            ranked_docs = self.rank_documents(query)
            results[query_id] = {doc_id: score for doc_id, score in ranked_docs[:top_k]}
        return results    

    def rank_documents(self, query_terms):
        """
        Rank documents according to their relevance to a given set of query terms using BM25
        """
        scores = {}
        for term in query_terms:
            #print(term)
            if term in self.inverted_index:
                for doc_id in self.inverted_index[term]:
                    #print(doc_id)
                    if doc_id not in scores:
                        scores[doc_id] = 0
                    scores[doc_id] += self.bm25_score(doc_id, query_terms)
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)

def normalize_scores(ranked_docs):
        if not ranked_docs:
            return []
        max_score = max(score for _, score in ranked_docs)
        min_score = min(score for _, score in ranked_docs)
        if max_score == min_score:
            return [(doc_id, 1.0) for doc_id, _ in ranked_docs]
        return [(doc_id, (score - min_score) / (max_score - min_score)) for doc_id, score in ranked_docs]