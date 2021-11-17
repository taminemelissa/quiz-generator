import tensorflow as tf
from functools import partial




#******************** global constants ******************

path = '/coding_linux20/programming/datasets/wikitext-2-raw'
AUTOTUNE = tf.data.AUTOTUNE




#********************* loading *************************

class DataLoader:

    def __init__(self, path: str) -> None:
        """[summary]

        Args:
            path ([type]): [description]
        """
        self.path = path
    
    
    def load(self) -> None:
        """[summary]
        """
    
        self.ds = tf.data.TextLineDataset(self.path)
        
    
    def prepare(self, lower = True, split = True) :
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
        """
    
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

        #*************basic treatment************
        
        ds = self.ds.map(lambda x: tf.strings.split(x, ' . '))
        ds = ds.map(all_mapping_in_one, num_parallel_calls = AUTOTUNE)

        ds = ds.unbatch().cache().prefetch(AUTOTUNE)

        ds = ds.filter(lambda x: tf.cast(tf.strings.length(x), bool))
        ds = ds.filter(lambda x: tf.cast(tf.strings.length(x)-1, bool))
        self.ds = ds.filter(content_filter)
        
        if lower:
            self.lower()
        
        if split:
            self.split()
        
        return self.ds
    
    
    def lower(self):
        
        self.ds = self.ds.map(lambda s : tf.strings.lower(s, encoding = 'utf-8'), num_parallel_calls = AUTOTUNE)
        
        
    def split(self):
        
        self.ds = self.ds.map(lambda s : tf.strings.split(s, sep = ' '), num_parallel_calls = AUTOTUNE)
    

