LANGUAGE = 'english'

PATH = '/coding_linux20/programming/datasets/wikitext-2-raw' + '/wiki.train.raw'

N_STRING_WORDCLOUD = 100

from nltk.corpus import stopwords
try:
    STOPWORDS = set(stopwords.words(LANGUAGE))
except LookupError:
    import nltk
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words(LANGUAGE))