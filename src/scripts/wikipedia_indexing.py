from typing import List
from os import listdir
from os.path import isfile, join
import json
from tqdm import tqdm
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def set_es_client():
    """Connection with ElasticSearch in the DataLab"""
    
    es = Elasticsearch([{'host': HOST, 'port': 9200}], http_compress=True,  timeout=200)
    return es


def convert_wikipedia_dump_to_documents(args, batch):
    """
    Convert Wikipedia dumps (text files) into dictionaries of documents composed of paragraphs
    :param args: Dictionary of arguments
    :param batch: Batch function
    :return: None
    """

    directory = args['directory']
    batch_size = args['batch_size']
    min_len_paragraph = args['min_len_paragraph']

    # Get all directions in the wikipedia folder
    wiki_dirs = [f for f in listdir(directory) if not isfile(join(directory, f))]
    dicts = []
    counts = dict(documents=0, paragraphs=0)
    progress_bar = tqdm(wiki_dirs)
    batch_id = 1

    for dirs in progress_bar:
        sub_dirs = [f for f in listdir(join(directory, dirs)) if not isfile(join(directory, dirs))]
        progress_bar.set_description(f"Processing wikipedia folder {dirs}")

        for file in sub_dirs:
            f = open(join(directory, dirs, file), "r")

            # Each text file contains json structures separated by one new-line character '\n'
            articles = f.read().split("\n")

            for article in articles:
                if len(article) == 0:
                    continue

                json_formatted_article = json.loads(article)
                base_document = {"id": json_formatted_article["id"],
                                 "name": json_formatted_article["title"],
                                 "url": json_formatted_article["url"]}
                counts["documents"] += 1

                paragraph_separator = '\n'  # Paragraphs in files are separated by one new-line character '\n'
                paragraphs = [p.strip() for pid, p in enumerate(json_formatted_article["text"].split(paragraph_separator))
                              if pid > 0 and p.strip() and len(p) >= min_len_paragraph]
                counts["paragraphs"] += len(paragraphs)

                for pid, p in enumerate(paragraphs):
                    document = {**base_document, "paragraph_id": pid, "text": p}
                    dicts.append(document)

                if len(dicts) >= batch_size:
                    batch(dicts, batch_id)
                    dicts = [] # Empty bulk
                    batch_id += 1

    # Process the last partial batch
    if dicts:
        batch(dicts, batch_id)

    logger.info("==" * 100)
    logger.info("Indexing done.")
    logger.info(f"# documents: {counts['documents']}")
    logger.info(f"# paragraphs: {counts['paragraphs']}, "
                    f"{counts['paragraphs'] / counts['documents']:.2f} per document")


def index_documents(client: Elasticsearch, index: str, documents: List[dict]):
    """
    Indexes documents for doing queries in Elasticsearch
    :param client: Elasticsearch client
    :param index: The index to write the documents into
    :param documents: List of dictionaries
    :return: None
    """
    documents_to_index = []

    for doc in documents:
        _doc = {"_op_type": "create", "_index": index, **doc}
        documents_to_index.append(_doc)

    bulk(client, documents_to_index, request_timeout=300)


def run_indexing(client: Elasticsearch, args):
    """
    Creates the final french wikipedia index
    :param client: Elasticsearch client
    :param args: Dictionary of arguments
    :return: None
    """

    index = "wikipedia"
    index += f"_{args['language'].lower()}"

    if client.indices.exists(index=index):
        logger.info(f'{index}')
        logger.warning(f"Index {index} already exists, deleting the index.")
        client.indices.delete(index=index)

    client.indices.create(index=index, body={
        "settings": {
            "analysis": {
                "analyzer": {
                    "default": {
                        "type": "standard",
                        "stopwords": "_english_"
                    }
                }
            }
        }
    }
                          )

    def batch(dicts, batch_id):
        index_documents(client, index, dicts)

    convert_wikipedia_dump_to_documents(args, batch=batch)