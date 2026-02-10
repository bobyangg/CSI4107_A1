#!/usr/bin/env python3
"""
trec_eval.py - A Python replacement for trec_eval that calculates standard IR evaluation metrics.
Compatible with TREC evaluation format.

Usage: python trec_eval.py qrels_file results_file

This script provides the same core functionality as the official trec_eval tool
but is implemented in Python for better cross-platform compatibility.
"""

import sys
from collections import defaultdict

def load_qrels(qrels_file):
    """
    Load relevance judgments from qrels file.
    Format: query_id iter doc_id relevance
    OR TSV format: query_id\tdoc_id\trelevance
    """
    qrels = defaultdict(set)
    with open(qrels_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
                
            # Skip header if present
            if line_num == 0 and line.startswith('query-id'):
                continue
                
            # Handle both space-separated and tab-separated formats
            if '\t' in line:
                parts = line.split('\t')
            else:
                parts = line.split()
                
            if len(parts) >= 3:
                query_id = parts[0].strip()
                doc_id = parts[-2].strip()  # doc_id is second-to-last in both formats
                relevance = int(parts[-1].strip())  # relevance is last
                
                if relevance > 0:  # Only store relevant documents
                    qrels[query_id].add(doc_id)
    
    return qrels

def load_results(results_file):
    """
    Load system results from TREC format file.
    Format: query_id Q0 doc_id rank score run_tag
    """
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

def calculate_metrics(qrels, results):
    """Calculate standard TREC evaluation metrics."""
    
    all_metrics = {}
    query_metrics = defaultdict(dict)
    
    total_queries = 0
    total_relevant = 0
    total_retrieved = 0
    total_rel_ret = 0
    
    sum_ap = 0.0
    sum_p5 = 0.0
    sum_p10 = 0.0
    sum_p30 = 0.0
    sum_recall = 0.0
    sum_rprec = 0.0
    sum_recip_rank = 0.0
    
    for query_id in sorted(qrels.keys()):
        if query_id not in results:
            continue
            
        relevant_docs = qrels[query_id]
        retrieved_docs = results[query_id]
        
        if not relevant_docs:
            continue
            
        total_queries += 1
        num_relevant = len(relevant_docs)
        num_retrieved = len(retrieved_docs)
        
        total_relevant += num_relevant
        total_retrieved += num_retrieved
        
        # Calculate metrics for this query
        rel_ret = 0
        precision_at_ranks = []
        recall_at_ranks = []
        
        for i, (doc_id, rank, score) in enumerate(retrieved_docs):
            if doc_id in relevant_docs:
                rel_ret += 1
            
            precision = rel_ret / (i + 1)
            recall = rel_ret / num_relevant
            
            precision_at_ranks.append(precision)
            recall_at_ranks.append(recall)
        
        total_rel_ret += rel_ret
        
        # Average Precision (AP)
        ap_sum = 0.0
        rel_found = 0
        for i, (doc_id, rank, score) in enumerate(retrieved_docs):
            if doc_id in relevant_docs:
                rel_found += 1
                ap_sum += rel_found / (i + 1)
        
        ap = ap_sum / num_relevant if num_relevant > 0 else 0.0
        sum_ap += ap
        
        # Precision at fixed ranks
        p5 = precision_at_ranks[4] if len(precision_at_ranks) > 4 else 0.0
        p10 = precision_at_ranks[9] if len(precision_at_ranks) > 9 else 0.0
        p30 = precision_at_ranks[29] if len(precision_at_ranks) > 29 else 0.0
        
        sum_p5 += p5
        sum_p10 += p10
        sum_p30 += p30
        
        # Recall
        final_recall = recall_at_ranks[-1] if recall_at_ranks else 0.0
        sum_recall += final_recall
        
        # R-precision (precision at R, where R is number of relevant docs)
        r_prec = 0.0
        if num_relevant <= len(retrieved_docs):
            r_retrieved = 0
            for i in range(min(num_relevant, len(retrieved_docs))):
                if retrieved_docs[i][0] in relevant_docs:
                    r_retrieved += 1
            r_prec = r_retrieved / num_relevant
        sum_rprec += r_prec
        
        # Reciprocal Rank (rank of first relevant document)
        recip_rank = 0.0
        for i, (doc_id, rank, score) in enumerate(retrieved_docs):
            if doc_id in relevant_docs:
                recip_rank = 1.0 / rank
                break
        sum_recip_rank += recip_rank
        
        # Store individual query metrics
        query_metrics[query_id] = {
            'ap': ap,
            'P_5': p5,
            'P_10': p10,
            'P_30': p30,
            'recall': final_recall,
            'Rprec': r_prec,
            'recip_rank': recip_rank,
            'num_rel': num_relevant,
            'num_ret': num_retrieved,
            'num_rel_ret': rel_ret
        }
    
    # Calculate overall metrics
    if total_queries > 0:
        all_metrics['map'] = sum_ap / total_queries
        all_metrics['P_5'] = sum_p5 / total_queries
        all_metrics['P_10'] = sum_p10 / total_queries
        all_metrics['P_30'] = sum_p30 / total_queries
        all_metrics['recall'] = sum_recall / total_queries
        all_metrics['Rprec'] = sum_rprec / total_queries
        all_metrics['recip_rank'] = sum_recip_rank / total_queries
        all_metrics['num_q'] = total_queries
        all_metrics['num_rel'] = total_relevant
        all_metrics['num_ret'] = total_retrieved
        all_metrics['num_rel_ret'] = total_rel_ret
    
    return all_metrics, query_metrics

def print_results(all_metrics, query_metrics):
    """Print results in trec_eval format."""
    
    # Print per-query results
    for query_id in sorted(query_metrics.keys()):
        metrics = query_metrics[query_id]
        print(f"map                  \t{query_id}\t{metrics['ap']:.4f}")
    
    for query_id in sorted(query_metrics.keys()):
        metrics = query_metrics[query_id]
        print(f"Rprec                \t{query_id}\t{metrics['Rprec']:.4f}")
    
    for query_id in sorted(query_metrics.keys()):
        metrics = query_metrics[query_id]
        print(f"P_5                  \t{query_id}\t{metrics['P_5']:.4f}")
    
    for query_id in sorted(query_metrics.keys()):
        metrics = query_metrics[query_id]
        print(f"P_10                 \t{query_id}\t{metrics['P_10']:.4f}")
    
    for query_id in sorted(query_metrics.keys()):
        metrics = query_metrics[query_id]
        print(f"P_30                 \t{query_id}\t{metrics['P_30']:.4f}")
    
    for query_id in sorted(query_metrics.keys()):
        metrics = query_metrics[query_id]
        print(f"recall               \t{query_id}\t{metrics['recall']:.4f}")
    
    for query_id in sorted(query_metrics.keys()):
        metrics = query_metrics[query_id]
        print(f"recip_rank           \t{query_id}\t{metrics['recip_rank']:.4f}")
    
    # Print overall averages
    print(f"map                  \tall\t{all_metrics['map']:.4f}")
    print(f"Rprec                \tall\t{all_metrics['Rprec']:.4f}")
    print(f"P_5                  \tall\t{all_metrics['P_5']:.4f}")
    print(f"P_10                 \tall\t{all_metrics['P_10']:.4f}")
    print(f"P_30                 \tall\t{all_metrics['P_30']:.4f}")
    print(f"recall               \tall\t{all_metrics['recall']:.4f}")
    print(f"recip_rank           \tall\t{all_metrics['recip_rank']:.4f}")
    print(f"num_q                \tall\t{all_metrics['num_q']}")
    print(f"num_rel              \tall\t{all_metrics['num_rel']}")
    print(f"num_ret              \tall\t{all_metrics['num_ret']}")
    print(f"num_rel_ret          \tall\t{all_metrics['num_rel_ret']}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python trec_eval.py qrels_file results_file")
        print("Example: python trec_eval.py scifact/scifact/qrels/test.tsv Results_FullText.txt")
        sys.exit(1)
    
    qrels_file = sys.argv[1]
    results_file = sys.argv[2]
    
    try:
        qrels = load_qrels(qrels_file)
        results = load_results(results_file)
        
        all_metrics, query_metrics = calculate_metrics(qrels, results)
        print_results(all_metrics, query_metrics)
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
