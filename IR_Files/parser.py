import json
import os

def parse_document(document_line):
    """
    Parse a single JSON line as a document
    """
    doc = json.loads(document_line)
    parsed_doc = {
        'DOCNO': doc['_id'],
        'HEAD': doc.get('title', 'NO_TITLE'),
        'TEXT': doc.get('text', 'NO_TEXT'),
        'URL': doc.get('metadata', {}).get('url', 'NO_URL')
    }
    return parsed_doc

def parse_documents_from_file(file_path):
    """
    Read the JSON lines file and parse each document
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        parsed_docs = [parse_document(line) for line in file]
    return parsed_docs

def parse_query(query_line):
    """
    Parse a single JSON line as a query
    """
    query = json.loads(query_line)
    parsed_query = {
        'num': query['_id'],
        'title': query.get('text', 'NO_TEXT'),
        'query': query.get('metadata', {}).get('query', 'NO_QUERY'),
        'narrative': query.get('metadata', {}).get('narrative', 'NO_NARRATIVE'),
        'url': query.get('metadata', {}).get('url', 'NO_URL')
    }
    return parsed_query

def parse_queries_from_file(file_path):
    """
    Read the JSON lines file and parse each query
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        parsed_queries = [parse_query(line) for line in file]
    return parsed_queries


def parse_documents_from_folder(folder_path):
    """
    Read all files in the specified folder and parse documents from each file
    """
    all_documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            all_documents.extend(parse_documents_from_file(file_path))
    return all_documents
