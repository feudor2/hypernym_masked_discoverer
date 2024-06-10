from wordnet_helper import WordNetHelper
from gensim.models.fasttext import FastText

class VectorModel:
    """
    A class used to load a pretrained Fasttext model and predict k-nearest elements in its vector space
    ...

    Attributes
    ----------
    model : gensim.Fasttext
        a pretrained fasttext model from gensim library
    WN : WordNetHelper
        a supporting WordNet-based utility

    Methods
    -------
    predict(set_, filter_=False, k=15)
        Prints the animals name and what sound it makes
    """

    def __init__(self, model_path):
        self.model = FastText.load(model_path)
        self.WN = WordNetHelper()
        
    def predict(self, dataset, k=15, filter_=True, col_in='data', col_out='pred'):
        dataset[col_out] =  self.get_nearest_from(dataset[col_in], filter_=filter_, k=k)
        return dataset
    
    def get_nearest_from(self, set_, filter_=False, k=15):
        """Gets topk nearest neighbours for all words in a given dataset

        Parameters
        ----------
        set_ : pandas.Series
            a data column from your dataset with words
        filter_ : bool, optional
            pass True to check if word is in WordNet corpus
        k : int, optional
            k nearest elements

        Returns
        -------
        list
            a list closest elements for every word
        """
        wv = self.model.wv
        if filter_:
            return [self.WN.check([x[0] for x in wv.most_similar(word, topn=k)]) for word in set_]
        return [','.join([x[0] for x in wv.most_similar(word, topn=k)]) for word in set_]