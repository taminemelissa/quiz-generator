import tensorflow as tf





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
    
        self.ds = tf.data.TextLineDataset(path)
        
    
    def prepare(self) -> tf.FilterDataset:
        """
        TODO Fonction à adapter en fonction de la dataset, marche ici pour une ds wikipédia classique
        Enleve les titres de paragraphes wikipedia
        split les strings
        enleve le symbole @-@ qui est un peu partout, surement les hyperliens je pense, mais qui ous est inutile
        enleve les strings vides
        enleve les strings d'un espace (transition entre paragraphe)
        cache et prefetch pour plus de rapidité
        """
    
        def content_filter(x: tf.Tensor) -> tf.Tensor:
            
            return tf.logical_not(tf.strings.regex_full_match(x, '([[:space:]][=])+.+([[:space:]][=])+[[:space:]]*'))
            
        def remove_the_at(x: tf.Tensor) -> tf.Tensor:
            
            return tf.strings.regex_replace(x, '@-@', '', replace_global=True)
            
        ds = self.ds.filter(content_filter)
        ds = ds.map(lambda x: tf.strings.split(x, ' . '))
        ds = ds.map(remove_the_at)
        ds = ds.unbatch()
        ds = ds.filter(lambda x: tf.cast(tf.strings.length(x), bool))
        self.ds = ds.filter(lambda x: tf.cast(tf.strings.length(x)-1, bool))
        
        self.ds.cache().prefetch(AUTOTUNE)

        return self.ds
    


