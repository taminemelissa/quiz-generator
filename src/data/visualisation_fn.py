# TODO : Write the summaries for each function
# TODO : We load from pickle files, perhaps it will change
import nltk
from nltk.corpus import stopwords
    
from typing import List, Any

import pickle

try:
    import constants as c
except ImportError:
    c = None
    raise ImportError('constants' + ' not imported')

try:
    STOPWORDS = set(stopwords.words(c.LANGUAGE))
except LookupError:
    import nltk
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words(c.LANGUAGE))





# ********************* loading *************************

class TextProcessing:
    
    def __init__(self, path: str) -> None:
        """[summary]
        
        Args:
            path ([type]): [description]
        """
        self.path = path
    
    
    def load(self) -> None:
        """
        load the sentences in a list of sequences
        """
    
        self.sequences = []
        with open('donnees_test', 'rb') as f:
            while True:
                try:
                    o = pickle.load(f)
                except EOFError:
                    break
                finally :
                    print('extraction done')
                self.sequences.append(o)
                
                
    def combine_first_strings(self, n_strings: int) -> str:
        """[summary]
        """
        combined_strings = ' '.join([s.numpy().decode('utf-8') for s in self.sequences[:]])

        return combined_strings
        

    def prepare(self, lower: bool = True, split: bool = True) -> str:
        
