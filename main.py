import os
import pickle
import pandas as pd

import yadisk
import zipfile
import nltk

from for_masked_lm import HypernymMaskedModel, ResultsWrapper
from vector_model import VectorModel
from prompt_utils import basic_prompts

if __name__ == 'main':
    # loading dataset
    with open('dataset.pickle', 'rb') as _:
        dataset = pickle.load(_)
    
    # download wordnet corpus
    nltk.download('wordnet')
    
    # load vector model (fasttext)
    y = yadisk.YaDisk()
    filename = 'fasttext.zip'
    url = 'https://disk.yandex.ru/d/dM3Vn2mlExzyZQ'
    y.download_public(url, filename)
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall()
    fasttext = VectorModel('fasttext/umbc_model_best.bin')
    
    # alternatively, you can load already predicted candidates
    '''with open('fasttext_predictions.pickle', 'rb') as _:
        preds = pickle.load(_)
    dataset['pred'] = preds'''
    
    # define the subset
    subset_name = '1A.english.trial'
    subset = dataset[dataset['set'] == subset_name]
    
    #if you are using vector model, get predictions 
    subset = fasttext.predict(subset, topk=15, filter_=True, col_in='data', col_out='pred')
    
    #loading model
    model = HypernymMaskedModel(model_path="bert-base-cased", device='cpu')
    with open('stopwords.pickle', 'rb') as _:
        STOPWORDS = pickle.load(_)
    
    # use model to get predictions
    output = model.prediction_in_dataset(
        subset,
        hyper_prompts=basic_prompts['hyper_prompts'],
        cohypo_prompts=basic_prompts['best_cohypo_prompt'],
        mixed_prompts=basic_prompts['mixed_prompts'],
        candidates=subset['pred'],
        k_out=15,
        k_hyper=15,
        k_hypo=15,
        col_in='data',
        stopwords=STOPWORDS,
        article=(True, True, False),
        seed=42
    )
    
    # unpack results
    results = ResultsWrapper(subset, tables=output, col_in='data')
    results.save()
    
    # getting metric values
    evaluation = results.calculate_metrics(col_gold='gold')
    display(evaluation)
    #evaluation.to_csv('bert_trial_results.tsv', sep='\t')
    