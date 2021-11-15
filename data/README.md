## French Wikipedia Dump

We download the latest French Wikipedia dump and extract its text using WikiExtractor in order to use ElasticSearch on it. 

```
# Get the latest French Wikipedia dump
wget "http://download.wikimedia.org/frwiki/latest/frwiki-latest-pages-articles.xml.bz2"

# Extract its text using WikiExtractor
python -m wikiextractor.WikiExtractor -o "quiz-generator/data/wikipedia/" --json --processes 12 "quiz-generator/data/wikipedia/frwiki-latest-pages-articles.xml.bz2"

# Remove the compressed file
rm "quiz-generator/data/wikipedia/frwiki-latest-pages-articles.xml.bz2"
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
        └── CG
```

