# TODO : We load from pickle files, perhaps it will change
import nltk
from nltk.corpus import stopwords
    
from typing import List, Any, Dict

import pickle

import re

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
    """[summary]
    In what follows:
    
    self.sequences is a list of sequences
    self.text is the whole concatenated text
    self.text_split is the list of the split text
    """
    
    def __init__(self, path: str) -> None:
        """[summary]
        
        Args:
            path ([type]): [description]
        """
        
        self.path = path
    
    
    def load(self) -> None:
        """[summary]
        Loads everything in the pickle file
        """
        
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
        

    def prepare(self, lower: bool = True, split: bool = True) -> None:
        """[summary]
        Gets the text ready to input it in the next functions
        Removes the first and last spaces
        It can create the following attributes, defined above the __init__:
        self.text
        self.text_split
        
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
        
        self.text_split = self.text.split()
    
    
    def without_stopwords(self) -> List[str]:
        """[summary]

        Returns:
            List[str]: list of words that are not stop words
        """
        
        if not hasattr(self, 'text'):
            self.prepare() #by default it creates text_split
            
        self.word_list_without_stop_words = [w for w in self.text_split if w not in STOPWORDS]
        
        return self.word_list_without_stop_words
    
    
    def without_stopwords_concatenated(self) -> str:
        """[summary]

        Returns:
            str: string of the words that are not stopwords
        """
        
        if not hasattr(self, 'word_list_without_stop_words'):
            self.without_stopwords() 

        self.text_without_stopwords = ' '.join(self.word_list_without_stop_words)
        
        return self.text_without_stopwords



def cardinality_of_words(l: List[str]) -> Dict[str, int]:
    """[summary]

    Args:
        l (List[str]): list of words

    Returns:
        Dict[str, int]: dictionary of the number of occurences per word in the list of words
    """
    
    cardinalities = {}
    
    for word in l:
        if word not in cardinalities:
            cardinalities[word] = 1
        else:
            cardinalities[word] += 1
    
    sorted_cardinalities = {k: v for k, v in sorted(cardinalities.items(), key = lambda item: item[1], reverse = True)}
    
    return sorted_cardinalities