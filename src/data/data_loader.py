import tensorflow as tf
import numpy as np

import nltk

try:
    import constants as c
except ImportError:
    c = None
    raise ImportError('constants' + ' not imported')



# ******************** global constants ******************

AUTOTUNE = tf.data.AUTOTUNE





# ********************* loading *************************

class DataLoader:

    def __init__(self, path: str) -> None:
        """[summary]
        the datasets are always in bytes, but sometimes the lists returned are in string like combine_first_strings
        
        Args:
            path ([type]): [description]
            language ([type]): ex : 'english'
        """
        self.path = path
        
        
    def load(self) -> None:
        """
        self.ds : the dataset, can be lowered, but never split
            tf.Tensor: shape=(x,), dtype=string, where x varies
            
        self.ds_split : the dataset split
            tf.Tensor: shape=(), dtype=string
            
        """
    
        self.ds = tf.data.TextLineDataset(self.path)
        
        self.ds_split = None
        
    
    def prepare(self, lower = True, split = True):
        """
        TODO faire en sorte qu'il y ait un seul mapping au lieu de 3 -> plus vite
        TODO traduire en anglais
        TODO Fonction à adapter en fonction de la dataset, marche ici pour une ds wikipédia classique
        split les strings
        fait le mapping des remplacements ie :
            enleve le symbole @-@ qui est un peu partout, surement les hyperliens je pense, mais qui ous est inutile
            eleve ce qui n'est pas une lettre (symboles chinois etc)
            enleve les espaces multiples et les remplace par un seul espace
        filtre les strings vides
        filtre les strings d'un caractère (transition entre paragraphe)
        filtre les titres de paragraphes wikipedia

        cache et prefetch pour plus de rapidité

        
        remove_symbols is equivalent of 'isalpha()'
        
        Args:
            lower (bool, optional): [description]. Defaults to True.
            split (bool, optional): [description]. Defaults to True.
        """
        
        if not hasattr(self, 'ds'):
            self.load()
    
        def content_filter(x: tf.Tensor) -> tf.Tensor:

            return tf.logical_not(tf.strings.regex_full_match(x, '([[:space:]][=])+.+([[:space:]][=])+[[:space:]]*'))

        def remove_the_at(x: tf.Tensor) -> tf.Tensor:

            return tf.strings.regex_replace(x, '@-@', '', replace_global=True)

        def remove_symbols(x):

            return tf.strings.regex_replace(x, '[^a-zA-Z ]', '', replace_global=True)

        def remove_multiple_spaces(x):

            return tf.strings.regex_replace(x, ' +', ' ', replace_global=True)
        
        def remove_first_and_last_spaces(x):
            
            return tf.strings.regex_replace(x, "^\s*|\s*$", '', replace_global=True)

        def all_mapping_in_one(x):
            
            return remove_first_and_last_spaces(remove_multiple_spaces(remove_symbols(remove_the_at(x))))

        # ************* basic treatment ************
        
        ds = self.ds.map(lambda x: tf.strings.split(x, ' . '))
        ds = ds.map(all_mapping_in_one, num_parallel_calls = AUTOTUNE)

        ds = ds.unbatch().cache().prefetch(AUTOTUNE)

        ds = ds.filter(lambda x: tf.cast(tf.strings.length(x), bool))
        ds = ds.filter(lambda x: tf.cast(tf.strings.length(x)-1, bool))
        self.ds = ds.filter(content_filter)
        
        # ************* additional treatment ************
        
        if lower:
            self.lower()
                
        if split:
            self.split()
                
        return self.ds_split if self.ds_split is not None else self.ds
    
    
    def lower(self):
        
        self.ds = self.ds.map(lambda s : tf.strings.lower(s, encoding = 'utf-8'), num_parallel_calls = AUTOTUNE)
        
        
    def split(self):
        
        self.ds_split = self.ds.map(lambda s : tf.strings.split(s, sep = ' '), num_parallel_calls = AUTOTUNE)
    
    def without_stopwords(self, n_strings):
        """when iterating over tf.Dataset, you can't iterate over one element with a comprehension list or tf.map_fn
        to remove the stopwords, we have to do it from the combined_strings

        Args:
            n_strings ([type]): [description]
        Returns:
            string
        """     
        combined_strings = self.combine_first_strings(n_strings)
        word_list = combined_strings.split()
        word_list_without_stop_words = [w for w in word_list if w not in c.STOPWORDS]
        
        return word_list_without_stop_words    

    def combine_first_strings(self, n_strings):
        """useful for wordcloud

        Args:
            n_strings ([type]): [description]
        Returns:
            string
        """
        combined_strings = ' '.join([s.numpy().decode('utf-8') for s in self.ds.take(n_strings).__iter__()])
        
        return combined_strings
    
    def combine_first_strings_all_ds(self, n_strings):
        """[summary]

        Args:
            n_strings ([type]): [description]
        """
        combined_strings = self.ds.reduce('', lambda s, s_: s + ' ' + s_)
        
        return combined_strings
    
    def combine_first_strings_without_stopwords(self, n_strings):
        """[summary]

        Args:
            n_strings ([type]): [description]
        """
        if not hasattr(self, 'ds_without_stopwords'):
            self.without_stopwords()
        
        combined_strings = ' '.join([s.numpy().decode('utf-8') for s in self.ds_without_stopwords.take(n_strings).__iter__()])
        
        return combined_strings