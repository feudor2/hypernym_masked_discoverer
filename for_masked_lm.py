import math
import random
import re
import os
from collections import defaultdict as defdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from wordnet_helper import WordNetHelper
from metrics import *
from prompt_utils import *


class HypernymMaskedModel:
    """
    A class used to predict hypernyms for words in a dataset using LLM and Fasttext model output
    ...

    Attributes
    ----------
    model : transformers.AutoModelForMaskedLM
        a model for filling mask
    tokenizer : transformers.Tokenizer
        tokenizer for masked model
    device : str, default 'cpu'
        device where to do the tensor calculations
    WN : WordNetHelper
        a supporting WordNet-based utility

    Methods
    -------
    set_random_seed(seed) : static
        use it to get reproducible results
    probable_words(prompt, k=15) :
        use to fill the mask in prompt with top 15 tokens
    find_word_probability(prompt, word) :
        calculate the total probability of all word's tokens to fill the mask in prompt
    predict_iterative(word, hyper_prompts, cohypo_prompts=None, mixed_prompts=None, candidates=None, k_out=15, k_hyper=15, k_hypo=15, stopwords=None) :
        main function to test different inference conditions
    prediction_in_dataset(word, hyper_prompts, cohypo_prompts=None, mixed_prompts=None, candidates=None, k_out=15, k_hyper=15, k_hypo=15, stopwords=None, col_in='data', seed=42) :
            use this function to get results from the model and to fix the random seed
    create_table_from_results(tables):
        an utility function to change the output list structure
    """
    def __init__(self, model_path="bert-base-cased", device='cpu'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForMaskedLM.from_pretrained(model_path).to(device)
        self.WN = WordNetHelper()
    
    @staticmethod
    def set_random_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        
    def probable_words(self, prompt, k=15):
        """Gets top k most probable words in the place of the mask

        Parameters
        ----------
        prompt : str
            a prompt with `_` to be replaced by mask (prompt must contain exactly one gap)
        k : int, optional
            number of tokens to extract from the model output

        Returns
        -------
        list
            a tuple of tokens and their probabilities (proper, not logarithmic)
        """
        if prompt.count("_") != 1:
            raise ValueError("The prompt must contain exactly one gap.")
        masked_prompt = prompt.replace("_", self.tokenizer.mask_token)
        tokenization = self.tokenizer(masked_prompt)["input_ids"]
        index = tokenization.index(self.tokenizer.mask_token_id)
        # tensor: 1 * L
        batch = torch.LongTensor([tokenization]).to(self.device)
        with torch.no_grad():
            logits = self.model(batch)["logits"][0]
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indexes = torch.topk(probs[index], dim=-1, k=k)
        return [w.replace('▁', '') for w in self.tokenizer.convert_ids_to_tokens(top_indexes)], top_probs.detach().cpu().numpy()
    
    def find_word_probability(self, prompt, word):
        """Gets probability of a word to replace the mask

        Parameters
        ----------
        prompt : str
            a prompt with `_` to be replaced by mask (prompt must contain exactly one gap)
        word : str
            a word to replace the mask; if it is multitoken, the total prob will be calculated

        Returns
        -------
        float
            the word's probability to occur in the prompt (proper, not logarithmic)
        """
        if prompt.count("_") != 1:
            raise ValueError("The prompt must contain exactly one gap.")
        masked_prompt = prompt.replace("_", self.tokenizer.mask_token)
        masked_tokenization = self.tokenizer(masked_prompt)["input_ids"]
        word_tokenization = self.tokenizer(word, add_special_tokens=False)["input_ids"]
        word_length = len(word_tokenization)
        index = masked_tokenization.index(self.tokenizer.mask_token_id)
        masked_tokenization[index:index+1] = [self.tokenizer.mask_token_id] * word_length
        batch = np.array([masked_tokenization] * word_length, dtype=int)
        for prefix_length in range(1, word_length):
            '''[
                [active MASK(x0) MASK MASK control]
                [active x0 MASK(x1) MASK control]
                [active x0 x1 MASK(x3) control]
            ]'''
            batch[prefix_length, index:index+prefix_length] = word_tokenization[:prefix_length]
        batch = torch.LongTensor(batch).to(self.device)
        with torch.no_grad():
            logits = self.model(batch)["logits"]
        log_probs = torch.log_softmax(
            logits[np.arange(word_length),index+np.arange(word_length)], dim=-1
        ).cpu().numpy()
        #print(log_probs.shape)
        subtoken_log_probs = log_probs[np.arange(word_length), word_tokenization]
        total_prob = subtoken_log_probs.sum()
        return np.exp(total_prob)
    
    def predict_iterative(
        self,
        word,
        hyper_prompts,
        cohypo_prompts=None,
        mixed_prompts=None,
        candidates=None,
        k_out=15,
        k_hyper=15,
        k_hypo=15,
        stopwords=None,
        article=True
    ):
        """A function to test different methods of hypernym prediction. The types and number of methods is currently not easy-modifiable

        Parameters
        ----------
        prompt : str
            a prompt with `_` to be replaced by mask (prompt must contain exactly one gap)
        word : str
            a target word to predict hypernyms for
        hyper_prompts : list
            list of prompts used for hypernym prediction (use `♥` for target word)
        cohypo_prompts : list, optional
            list of prompts used for co-hyponym prediction (use `♥` for target word)
        mixed_prompts : list, optional
            list of prompts used for hypernym prediction with extra co-hyponym (use `♥` for target word and `♠` for co-hypo)
        candidates : list, optional
            list of candidates to identify as hypernyms or co-hyponyms
        k_out : int, optional
            number of best candidates to return
        k_hyper : int, optional
            number of best hypernyms to predict
        k_hypo : int, optional
            number of best co-hyponyms to predict
        stopwords : list, optional
            words not to be included in the final list
        article : bool, optional
            if item is True, an appropriete undefinite article will be added to prompts (hyper, mixed, cohypo)
            
        Returns
        -------
        list
            list of discovered candidates for the word within each prompt
        """
        def divide_classes(hypers, hypos):
            '''Function to identify input candidates as hypernyms or co-hyponyms based on their model probability'''
            hypo_probs = np.max(
                [[c[j][1] for c in hypos] for j in range(len(hypos[0]))], axis=1
            )
            hyper_out, hypo_out = [], []
            for hyper in hypers:  # step = prompt
                diff = [
                    (c_hyper[0], c_hyper[1] - hypo_probs[j])
                    for j, c_hyper in enumerate(hyper)
                ]
                hyper_out.append(
                    [c[0] for j, c in enumerate(diff) if c[1] > 0]
                )  # and c[1] >= p_th_hyper[j]])
                hypo_out.append(
                    [c[0] for j, c in enumerate(diff) if c[1] <= 0]
                )  # and c[1] >= p_th_hypo[j]])
            return [[c for c in pr if c[0] in hyper_out[i]] for i, pr in enumerate(hypers)], [[c for c in pr if c[0] in hypo_out[i]] for i, pr in enumerate(hypos)]

        def merge_lists(*lists):
            '''An utility to merge lists with candidates'''
            out = []
            for i in range(len(lists[0])):
                out.append(lists[0][i])
                for l in lists[1:]:
                    out[-1] += l[i]
            return out

        def best_cohypo(preds):
            '''A function use to find the most suitable co-hyponym for the target word'''
            # preds = [(tokens_0, probs_0), ... (tokens_m, probs_m)]
            scores = defdict(int)
            for x in preds:
                for t, p in zip(x[0], x[1]):
                    scores[t] += p
            return max(scores, key=scores.get)

        def rearrange(candidates_, with_prob=True):
            '''A function to rearrange list of candidates based on their predicted probability'''
            out = []
            for pr in candidates_:
                out.append([c if with_prob else c[0] for c in sorted(pr, reverse=True, key=lambda x: x[1])])
            return out

        def rebuild(candidates_, mode='outer'):
            '''An utility funtion to reorganize list of candidates'''
            if mode == 'inner':
                return [[(c, p) for x in prompt for c, p in zip(*x)] for prompt in candidates_]
            return [[(c, p) for c, p in zip(*prompt)] for prompt in candidates_]

        def export_format(results_, k, i=0, reserve=None):
            '''A function to prepare list of candidates for export (filling & truncation)'''
            if not results_ and reserve is not None:
                return [[c[0] for c in pr][:k] for pr in reserve[i]]
            if reserve is not None:
                return [[c[0] for c in pr][:k] if pr else [c[0] for c in reserve[i][j]][:k] for j, pr in enumerate(results_)]
            return [[c[0] for c in pr][:k] for pr in results_]

        def wn_filter(results_, word, mode='all', stopwords=None, min_length=2):
            '''A function to filter out non-word candidates'''
            def unique(candidates_):
                out = []
                lemmas = [c[0] for c in candidates_]
                for i, c in enumerate(candidates_):
                    if not lemmas[i].isdigit() and len(lemmas[i]) > min_length and lemmas[i] not in lemmas[:i] and (stopwords is None or lemmas[i] not in stopwords):
                        out.append(c)
                return out
            if mode == 'single':
                return [self.WN.lemmatize(c) for c in results_[0] if self.WN.check(self.WN.lemmatize(c)) and self.WN.lemmatize(c) != word and (stopwords is None or c not in stopwords)], results_[1]
            return [unique([((c[0]), c[1]) for c in pr if self.WN.check(self.WN.lemmatize(c[0])) and self.WN.lemmatize(c[0]) != word])
                    for pr in results_]

        def from_cohypo(cohypos):
            '''A function to predict new hypernyms for discovered co-hyponyms'''
            to_hyper = [[wn_filter(self.probable_words(prepare_prompts(c[0], hyper_prompts)[i], k=k_hyper, article=article[0]),
                                   word=c[0], mode='single', stopwords=stopwords)
                        for c in pr]
                        for i, pr in enumerate(cohypos)]
            to_hyper = rebuild(to_hyper, mode='inner')
            ranged_to_hyper = []
            for pr in to_hyper:
                ranged_to_hyper.append(defdict(list))
                for c in pr:
                    ranged_to_hyper[-1][c[0]].append(c[1])
            return rearrange([[(c, np.mean(probs)) for c, probs in pr.items()] for pr in ranged_to_hyper])


        def iterative(word, candidates_, prompts, cohypo, MAX=1.0):
            '''A function to iteratively rearrange the list of predicted hypernyms'''
            out = []
            for i, cs in enumerate(candidates_):
                if not cs:
                    out.append([])
                    continue
                max_score = 0.0
                seq = cs
                word_i = seq[0]
                selected = [word_i[0]]
                prompt_i = prepare_prompts(word_i[0], prompts, cohypo=cohypo, article=article[0])[i]
                scores = {c[0]:[c[1]] for c in seq}
                while True:  
                    seq_new = [(c[0], self.find_word_probability(prompt_i, c[0])) for c in seq if c[0] not in selected]
                    if not seq_new:
                        break
                    seq_new = rearrange([seq_new + [(x, MAX) for x in selected]])[0]
                    word_i = [x for x in seq_new if x[0] not in selected][0]
                    new_score = word_i[1]
                    if max_score >= new_score:
                        break
                    max_score = new_score
                    selected.append(word_i[0])
                    prompt_i = prepare_prompts(word_i[0], prompts, cohypo=cohypo, article=article[0])[i]
                    seq = seq_new
                    for c in seq_new:
                        scores[c[0]].append(c[1])
                scores = rearrange([[(w, np.mean(sc)) for w, sc in scores.items()]])[0]
                out.append(scores)
                #print('#'*25)
            return out

        results_hyper = [
            self.probable_words(p, k=k_hyper)
            for p in prepare_prompts(word, hyper_prompts, article=article[0])
        ]
        results_hyper = wn_filter(rebuild(results_hyper), word=word, stopwords=stopwords)
        iter_hyper = iterative(word, results_hyper, hyper_prompts, '', MAX=1.0)
        results_mixed = []
        cohypo_best = None
        if cohypo_prompts:
            cohypos = [
            self.probable_words(p, k=k_hypo)
            for p in prepare_prompts(word, cohypo_prompts, article=article[2])
        ]      
            cohypo_best = best_cohypo(cohypos)

            if mixed_prompts:
                results_mixed = [
                    self.probable_words(p, k=k_hyper)
                    for p in prepare_prompts(word, mixed_prompts, cohypo=cohypo_best, article=article[1])
                ]
                results_mixed = wn_filter(rebuild(results_mixed), word=word, stopwords=stopwords)
                iter_mixed = iterative(word, results_mixed, mixed_prompts, '', MAX=1.0)
            else:
                results_mixed, iter_mixed = [], []

        if not isinstance(candidates, float):
            if isinstance(candidates, str):
                candidates = candidates.split(",")
            else:
                raise TypeError('Candidates must be str separated by `,`.')
            candidate_hyper = [
                [(c, self.find_word_probability(p, c,)) for c in candidates]
                for p in prepare_prompts(word, hyper_prompts, article=article[0])
            ]
            candidate_cohypo = [
                [(c, self.find_word_probability(p, c)) for c in candidates]
                    for p in prepare_prompts(word, cohypo_prompts, article=article[2])]

            candidates_hyper, candidates_cohypo = divide_classes(
                candidate_hyper,
                candidate_cohypo
            )
            if candidates_cohypo != [[], []]:
                results_ft_cohypo = from_cohypo(candidates_cohypo)
                #print(candidates_cohypo, results_ft_cohypo, sep='\n')
                iter_ft_cohypo = iterative(word, results_ft_cohypo, hyper_prompts, '', MAX=1.0)
            else:
                results_ft_cohypo = []
                iter_ft_cohypo = []

            if mixed_prompts:
                candidate_mixed = [
                    [(c, self.find_word_probability(p, c)) for c in candidates]
                    for p in prepare_prompts(word, mixed_prompts, cohypo=cohypo_best, article=article[1])
                ]
                candidates_mixed, candidates_mixed_cohypo = divide_classes(
                    candidate_mixed,
                    candidate_cohypo
                    )
                if candidates_mixed_cohypo != [[], []]:
                    results_ft_cohypo_mixed = from_cohypo(candidates_mixed_cohypo)
                    iter_ft_cohypo_mixed = iterative(word, results_ft_cohypo_mixed, mixed_prompts, cohypo=cohypo_best, MAX=1.0)
                else:
                    results_ft_cohypo_mixed = []
                    iter_ft_cohypo_mixed = []
                results_ft_mixed = merge_lists(candidates_mixed, results_mixed)
                iter_ft_mixed = iterative(word, merge_lists(results_mixed, candidates_mixed), mixed_prompts, cohypo=cohypo_best, MAX=1.0)
            else:
                results_ft_mixed, iter_ft_mixed = [], []
                results_ft_cohypo, results_ft_cohypo_mixed = [], []
                iter_ft_cohypo, iter_ft_cohypo_mixed = [], []

            results_ft_hyper = merge_lists(candidates_hyper, results_hyper)
            #ft_cohypos = merge_lists(candidates_hypo, cohypos)

            iter_ft_hyper = iterative(word, merge_lists(results_hyper, candidates_hyper), hyper_prompts, '', MAX=1.0)
        else:
            candidates = []
            results_ft_hyper, results_ft_mixed = [], []
            iter_ft_hyper, iter_ft_mixed = [], []
            results_ft_cohypo, results_ft_cohypo_mixed = [], []
            iter_ft_cohypo, iter_ft_cohypo_mixed = [], []

        to_export = [
            [results_hyper, results_mixed],
            [iter_hyper, iter_mixed],
            [results_ft_hyper, results_ft_mixed],
            [iter_ft_hyper, iter_ft_mixed],
            [results_ft_cohypo, results_ft_cohypo_mixed],
            [iter_ft_cohypo, iter_ft_cohypo_mixed],
        ]
        reserve = to_export[:2]
        return [[export_format(x, k_out, i, reserve[j % 2]) for i, x in enumerate(pair)] for j, pair in enumerate(to_export)]
   
    def create_table_from_results(self, data, hyper_prompts, mixed_prompts, cohypo_prompts, n_conditions=6):
        '''An utility function to change the output list structure'''
        # table format is `{prompts : [words from prompts]}`
        prompts = hyper_prompts + mixed_prompts
        out = []
        for i in range(n_conditions):
            data_i = data[i][0] + data[i][1]
            out.append({p: c for p, c in zip(prompts, data_i)})
        return out
    
    def prediction_in_dataset(
        self,
        dataset,
        hyper_prompts,
        cohypo_prompts=None,
        mixed_prompts=None,
        candidates=None,
        k_out=15,
        k_hyper=15,
        k_hypo=15,
        col_in='data',
        stopwords=None,
        article=(True, True, True),
        seed=42
    ):
        """A function to predict hypernyms for words in a dataset

        Parameters
        ----------
        dataset : pandas.DataFrame
            a dataframe containing `col_in` with input words data
        prompt : str
            a prompt with `_` to be replaced by mask (prompt must contain exactly one gap)
        word : str
            a target word to predict hypernyms for
        hyper_prompts : list
            list of prompts used for hypernym prediction (use `♥` for target word)
        cohypo_prompts : list, optional
            list of prompts used for co-hyponym prediction (use `♥` for target word)
        mixed_prompts : list, optional
            list of prompts used for hypernym prediction with extra co-hyponym (use `♥` for target word and `♠` for co-hypo)
        candidates : list, optional
            list of candidates to identify as hypernyms or co-hyponyms
        col_in : str, optional
            the dataset column, where all the input words data is stored
        k_out : int, optional
            number of best candidates to return
        k_hyper : int, optional
            number of best hypernyms to predict
        k_hypo : int, optional
            number of best co-hyponyms to predict
        stopwords : list, optional
            words not to be included in the final list
        article : tuple, optional
            if item is True, an appropriete undefinite article will be added to prompts (hyper, mixed, cohypo)
        seed : int, optional
            a seed to fix random states
            
        Returns
        -------
        list
            of packed data with predictions; use ResultsWrapper to unpack and calculate metrics
        """
        self.set_random_seed(seed)
        tables = [
            self.create_table_from_results(
                self.predict_iterative(
                    w,
                    hyper_prompts,
                    cohypo_prompts,
                    mixed_prompts,
                    candidates=c,
                    k_hyper=k_hyper,
                    k_hypo=k_hypo,
                    k_out=k_out,
                    stopwords=stopwords,
                    article=article
                ),
                normalize_prompts(prepare_prompts('<target>', hyper_prompts, article=article[0])),
                normalize_prompts(prepare_prompts('<target>', mixed_prompts, cohypo='<cohypo>', article=article[1])),
                normalize_prompts(prepare_prompts('<target>', cohypo_prompts, article=article[2])),
            )
            for w, c in tqdm(zip(dataset[col_in], candidates))
        ]
        return tables

class ResultsWrapper:
    """
    A class used to operate with HypernymMaskedModel output
    ...

    Attributes
    ----------
    dataset : pandas.DataFrame
        a dataframe containing `col_in` with input words data
    col_in : str, optional
        the dataset column, where all the input words data is stored
    tables : list, optional
        list, structured in HypernymMaskedModel output format; do not pass to load saved results
    conitions : list, optional
        names of methods of hypernym discovery

    Methods
    -------
    fix_tables(tables) : 
        an utility to check the fix the structure of tables
    compose_tables(tables) :
        convert tables to appropriate dataframe format
    save(path='', prefix='bert_results_') :
        save dataframes as tsv
    load(path='', prefix='bert_results_') :
        load previously saved dataframes with results
    results(sets_preds, model_name='Model', map_pref='', mrr_pref='', col1='gold', col2='pred', k=15) :
        used to calculate metrics and return the results as a dataframe
    assemble(main, add) : 
        an utility to simplify the concatenation
    get_results(self, tables, col_gold='gold') :
        operates with tables and returns the df with evaluation results
    calculate_metrics(col_gold='gold') :
        use this function to actually get the results from the ResultsWrapper instance
    """
    def __init__(self, dataset, tables=None, col_in='data', conditions=None):
        self.dataset = dataset
        self.col_in = col_in
        self.tables = self.compose_tables(self.fix_tables(tables)) if tables else []
        if not conditions:
            self.conditions = ['bert_hyper','bert_hyper_iter',
                               'ft+bert_hyper', 'ft+bert_hyper_iter',
                               'ft+bert_cohypo', 'ft+bert_cohypo_iter']
        
    def fix_tables(self, tables):
        '''Fix tables with missing values to build dataframes with no exceptions'''
        for w in tables:
            for i in range(1,6):
                missing = [p for p in w[0].keys() if p not in w[i].keys()]
                for p in missing:
                    w[i][p] = w[0][p]
        return tables
    
    def compose_tables(self, tables):
        '''An utility to convert model output lists to dataframes'''
        n = len(tables[0]) #word [ table
        output = []
        for i in range(n): #each training condition
            output.append(defdict(list))
            for t in tables: #each word
                for prompt, vals in t[i].items():
                    output[i][prompt].append(",".join(vals))
        output = [pd.concat([self.dataset[[self.col_in]].reset_index(drop=True), pd.DataFrame(t)], axis=1) for t in output]
        return output
    
    def save(self, path='', prefix='bert_results_'):
        '''Save processed tables with predicted hypernyms'''
        for i in range(len(self.conditions)):
            self.tables[i].to_csv(os.path.join(path, f'{prefix}{self.conditions[i]}.tsv'), sep='\t')
                                  
    def load(self, path='', prefix='bert_results_'):
        '''Load previously saved dataframes'''
        for cond in self.conditions:
            self.tables.append(pd.read_csv(os.path.join(path, f'{prefix}{cond}.tsv'), 
                                           index_col=0, sep='\t').drop(self.col_in, axis=1))
    
    def results(self, sets_preds, model_name='Model', map_pref='', mrr_pref='', col1='gold', col2='pred', k=15):
        '''A function to build the dataframe and evaluate model performance'''
        map_pref = f'MAP@{k}-' if not map_pref else map_pref
        mrr_pref = f'MRR@{k}-' if not mrr_pref else mrr_pref
        results = {map_pref+key:[np.round(map(value, col1, col2, k=k), 5)] for key, value in sets_preds.items()}
        results.update({mrr_pref+key:[np.round(mrr(value, col1, col2, k=k), 5)] for key, value in sets_preds.items()})
        return pd.DataFrame(results, index=[model_name])
    
    @staticmethod
    def assemble(main, add):
        '''A concatention shortcut'''
        return pd.concat([main, add], axis=1)
    
    def get_results(self, tables, col_gold='gold'):
        '''Process tables in a `for` loop'''
        results = []
        for col in self.tables[0].columns:
            results.append(
                self.results(
                    {self.conditions[i]: tables[i] for i in range(len(self.conditions))},
                    model_name=[col],
                    col1=col_gold,
                    col2=col,
                )
            )
        return pd.concat(results)
    
    def calculate_metrics(self, col_gold='gold'):
        '''A function to get evaluate predictions'''
        tables = [self.assemble(self.dataset.reset_index(drop=True), t) for t in self.tables]
        return self.get_results(tables, col_gold) 