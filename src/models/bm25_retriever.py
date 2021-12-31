from elasticsearch import Elasticsearch
from src.data.data_format import *


class BM25Retriever:

    def __init__(self, client, name: str = 'BM25 Retriever', **kwargs):
        """
        :param name: the name of the model
        :param kwargs: the arguments of the BM25 retriever
        """
        self.kwargs = self.fill_default_kwargs(**kwargs)
        self.name = name
        self.client = client

    @classmethod
    def fill_default_kwargs(cls, **kwargs) -> Dict:
        kwargs.update(dict(
            top_k=kwargs.get("top_k", 10),
            index=kwargs.get("index", "wikipedia_english")))
        return kwargs


    def convert_es_hit_to_context(self, hit: dict) -> Context:
        """
        Converts an Elasticsearch hit into a Context object
        :param hit: Elasticsearch hit
        :return: Context object containing the ES hit information
        """

        meta = {k: v for k, v in hit["_source"].items() if
                k not in ("text", "id")}
        meta["id"] = hit["_source"]["id"],
        meta["document_title"] = meta.pop("name", None)

        return Context(
            text=hit["_source"]["text"],
            title=meta['document_title'],
            identifier=hit["_id"],
            scores=OrderedDict({self.name: hit["_score"] if hit["_score"] else None}),
            meta=meta)

    def retrieve(self, query: str, top_k: int = 0) -> List[Context]:
        """
        Returns the top k passages in the index corresponding to the request
        :param query: A request for information about data in the Elasticsearch index
        :param top_k: k most relevant passages according to the query
        :return: A list containing the k most relevant Context
        """

        index = self.kwargs.get('index')
        if top_k == 0:
            top_k = self.kwargs.get('top_k')

        body = {
            "size": str(top_k),
            "query": {
                "bool": {
                    "should": [
                        {"multi_match": {"query": query,
                                         "type": "most_fields",
                                         "fields": "text"}}]
                }
            },
        }

        result = self.client.search(index=index, body=body)["hits"]["hits"]

        contexts = [self.convert_es_hit_to_context(hit) for hit in result]
        return contexts
