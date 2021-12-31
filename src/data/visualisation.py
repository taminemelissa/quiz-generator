from nltk.corpus import stopwords
from nltk import RegexpTokenizer

import spacy

import numpy as np

from sklearn.decomposition import PCA
    
from typing import List, Any, Dict

import pickle

import matplotlib.pyplot as plt

import re

from src.data.constants import *

try:
    STOPWORDS = set(stopwords.words(LANGUAGE))
except LookupError:
    import nltk
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words(LANGUAGE))

from src.data.data_format import *

from pywaffle import Waffle




# ********************* loading *************************

class TextProcessing:
    """
    In what follows:
    self.path is the path to a pickle file containing string sequences
    self.retrieved_contexts is a list of contexts instances defined in data_format
    
    self.sequences is a list of sequences
    self.text is the whole concatenated text
    self.text_split is the list of the split text.
    """
    
    def __init__(self, retrieved_contexts: List[Context] = None, path: str = None) -> None:
        """
        
        Args:
            path ([type]): path to pickle file.
        """
        
        self.path = path
        self.retrieved_contexts = retrieved_contexts
    
    
    def load(self) -> None:
        """
        Loads everything in the pickle file.
        """
        
        if self.path:

            self.sequences = []
            
            with open(self.path, 'rb') as f:
                while True:
                    try:
                        o = pickle.load(f)
                    except EOFError:
                        break
                    finally :
                        print('extraction done')
                    self.sequences.append(o)
        
        elif self.retrieved_contexts:

            self.sequences = [context.text for context in self.retrieved_contexts]


    def prepare(self, lower: bool = True, split: bool = True) -> None:
        """
        Gets the text ready to input it in the next functions
        Removes the first and last spaces
        It can create the following attributes, defined above the __init__:
        self.text
        self.text_split.
        
        Args:
            lower (bool, optional): whether to lower the text (create self.text) or not. Defaults to True.
            split (bool, optional): whether to split the text (create self.text_split) or not. Defaults to True.
        """
        
        if not hasattr(self, 'sequences'):
            self.load()

        self.text = ' '.join(self.sequences)
    
        expression_spaces = re.compile('^\s*|\s*$')
        
        def remove_first_and_last_spaces(s: str) -> str:
            
            return re.sub(expression_spaces, '', s)
        
        self.text = remove_first_and_last_spaces(self.text)
        
        if lower:
            self._lower()

        if split:
            self._split()
            
    
    def _lower(self) -> None:
        
        self.text = self.text.lower()
        
        
    def _split(self) -> None:
        """
        This tokenize makes it possible to split "l'école" into ["l'", "école"], and removes ponctuation by the way.
        """
        tokenizer = RegexpTokenizer(r'\w+')

        self.text_split = tokenizer.tokenize(self.text)

    
    def without_stopwords(self) -> List[str]:
        """

        Returns:
            List[str]: list of words that are not stop words.
        """
        
        if not hasattr(self, 'text'):
            self.prepare() #by default it creates text_split
            
        self.word_list_without_stop_words = [w for w in self.text_split if w not in STOPWORDS]
        
        return self.word_list_without_stop_words
    
    
    def without_stopwords_concatenated(self) -> str:
        """

        Returns:
            str: string of the words that are not stopwords.
        """
        
        if not hasattr(self, 'word_list_without_stop_words'):
            self.without_stopwords() 

        self.text_without_stopwords = ' '.join(self.word_list_without_stop_words)
        
        return self.text_without_stopwords



def cardinality_of_words(l: List[str]) -> Dict[str, int]:
    """

    Args:
        l (List[str]): list of words.

    Returns:
        Dict[str, int]: dictionary of the number of occurences per word in the list of words.
    """
    
    cardinalities = {}
    
    for word in l:
        if word not in cardinalities:
            cardinalities[word] = 1
        else:
            cardinalities[word] += 1
    
    sorted_cardinalities = {k: v for k, v in sorted(cardinalities.items(), key = lambda item: item[1], reverse = True)}
    
    return sorted_cardinalities



class Embedding:
    
    def __init__(self, name) -> None:
        """

        Args:
            name ([type]): name of the spacy model
        """
        
        self.model = spacy.load(name)
        
    
    def embedding(self, text: str) -> None:
        """

        Args:
            text ([type]): string containing the text
        """
        
        self.doc = self.model(text)
        self.words = [w.text for w in self.doc]
        self.words_embedding = [w.vector for w in self.doc]
    
    
    def pca(self) -> np.ndarray:
        
        if not hasattr(self, 'words_embedding'):
            raise NameError('words should be embedded first')
        
        pca = PCA(n_components=2)
        
        self.words_embedding_dim_2 = pca.fit_transform(self.words_embedding)
        
        return self.words_embedding_dim_2

    
def graph_occurrence(word, contexts):
    data = {}
    for i in range(len(contexts)):
        total_occurrences = contexts[i].text.lower().count(word)
        data[i] = total_occurrences
    fig = plt.figure(FigureClass=Waffle,rows=4,values=data,title={'label': 'Utilisation du mot "%s" dans les différents contextes' %word, 'loc': 'left'}, labels=["{0} ({1})".format(k, v) for k, v in data.items()])
    return fig