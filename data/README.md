## French Wikipedia Dump

We download the latest French Wikipedia dump and extract its text using WikiExtractor in order to use ElasticSearch on it. 

```
# Get the latest French Wikipedia dump
wget "http://download.wikimedia.org/frwiki/latest/frwiki-latest-pages-articles.xml.bz2"

# Extract its text using WikiExtractor
path/to/environment/with/wikiextractor/python -m wikiextractor.WikiExtractor -o "//data/wikipedia/" --json --filter_disambig_page --processes 12 "//data/wikipedia/frwiki-latest-pages-articles.xml.bz2"

# Remove the compressed file
rm "//data/wikipedia/frwiki-latest-pages-articles.xml.bz2"
```

The structure of the data folder looks like this :
```
├── data 
    ├── datasets
    ├── models
    └── wikipedia
        ├── AA
        ├── AB
        ├── ...
        ├── docs.db (wikipedia dump SQLite)
        └── docs-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz
```

