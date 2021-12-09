from elasticsearch import Elasticsearch
from src.data.data_format import *


class BM25Retriever:

    def __init__(self, name: str = 'BM25 Retriever', **kwargs):
        self.kwargs = self.fill_default_kwargs(**kwargs)
        self.name = name
        self._construct()

    @classmethod
    def fill_default_kwargs(cls, **kwargs) -> Dict:
        kwargs.update(dict(
            top_k=kwargs.get("top_k", 10),
            host=kwargs.get("host", "localhost"),
            index=kwargs.get("index", "wikipedia_french")))
        return kwargs

    def _construct(self):
        self.client = Elasticsearch(hosts=[{"host": self.kwargs.get("host"),
                                            "port": 9200}],
                                    http_compress=True,
                                    timeout=200)
        self.index = self.kwargs.get("index")

    def convert_es_hit_to_context(self, hit: dict) -> Context:
        """
        Converts an Elasticsearch hit into a Context object
        :param hit: Elasticsearch hit
        :return: Context object containing the ES hit information
        """

        meta = {k: v for k, v in hit["_source"].items() if
                k not in ("text", "external_source_id")}
        meta["external_source_id"] = hit["_source"]["external_source_id"],
        meta["document_title"] = meta.pop("name", None)

        return Context(
            text=hit["_source"]["text"],
            title=meta['document_title'],
            identifier=hit["_id"],
            scores=OrderedDict({self.name: hit["_score"] if hit["_score"] else None}),
            meta=meta)

    def retrieve(self, query: str, top_k: int = 0) -> List[Context]:
        """
        Returns the top k passages in the index corresponding to the query
        :param query: A request for information about data in the Elasticsearch index
        :param top_k: k most relevant passages according to the query
        :return: A list containing the k most relevant Context
        """

        index = self.index
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

