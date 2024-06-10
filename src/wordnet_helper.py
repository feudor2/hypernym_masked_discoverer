import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

class WordNetHelper:
    """
    An utility class based on nltk's WordNet subpackage
    ...

    Attributes
    ----------
    lemmas : list
        all lemmas from WordNet corpus
    lemmatizer : WordNetLemmatizer
        an instance of nltk's lemmatizer

    Methods
    -------
    check(words) :
        for every word in `words` check if it is in the dictionary
    lemmatize(word) :
        lemmatizes a word with lemmatizer
    """

    def __init__(self):
        #nltk.download('wordnet') # download it in the main.py
        self.lemmas = set(wn.all_lemma_names())
        self.lemmatizer = WordNetLemmatizer()
        
    def check(self, words):
        '''Check if `words` from list are in WordNet lemmas
        Return True if `words` is a single word in WN
        Return joined str if `words` is a str list'''
        
        def prettify(word):
            '''Lower the word, replace ' ' with '_' '''
            return word.lower().replace(' ', '_')
        
        if isinstance(words, list):
            return ','.join([word for word in words if prettify(word) in self.lemmas])
        if words is None:
            return ''
        return prettify(words) in self.lemmas
    
    def lemmatize(self, word):
        return self.lemmatizer.lemmatize(word)