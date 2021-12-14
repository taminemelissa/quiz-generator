## Simple English Wikipedia Dump

We download the latest Simple English Wikipedia dump and extract its text using WikiExtractor in order to use ElasticSearch on it. 

```
# Get the latest English Wikipedia dump
wget "http://download.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2"

# Extract its text using WikiExtractor
python -m wikiextractor.WikiExtractor -o "quiz-generator/data/wikipedia/" --json --processes 12 "quiz-generator/data/wikipedia/simplewiki-latest-pages-articles.xml.bz2"

# Remove the compressed file
rm "quiz-generator/data/wikipedia/simplewiki-latest-pages-articles.xml.bz2"
```

## English Stanza
We download the English version of Stanza.
```
import stanza
stanza.download('en', model_dir="quiz-generator/data/stanza")
```
## Final structure of the data folder
The structure of the data folder looks like this :
```
├── data 
    ├── stanza    
        ├── en
        ├── ressources.json
    └── wikipedia
        ├── AA
        ├── AB
        └── ...
```

