# A Masked LM Model for Hypernym Discovery
## Task description
Hypernym discovery is one of the most crucial natural languge processing task. It is in demand in automatic ontology learning and enrichment, query augmentation and to some extent in textual entailment and question answering.

Despite the fact that hyponym-hyperonym pairs occur in texts in special patterns like '<target> is a type of <hypernym>', for a certain word the corpus may not contain such a context at all. Another possible source of hypernymy knowledge are word2vec models. Those models learn vector representations of words based on surrounding context, so the words occurring in similar contexts have more similar vectors. A word can easily be replaced by its hypernyms, but not only hypernyms -- synonyms and other closely related words and associations can also infiltrate the closest candidates' list. Finally, like any other information that can be acquired from texts hypernymy may as well be revealed by transformer language models.

This project represents an attempt to unify all these sources and develop a fast and convenient approach to discover hypernyms for words. On one hand, the suggested model makes use of LM's ability to predict probability when filling the mask, that is integrated into the context pattern used as prompt. Thus it can generate the appropriate hypernyms for a term. On the other hand, the model tries to overcome the LM's limitation to predict only one token in the place of mask by engaging candidate terms from a vector model. Word2vec model output can be multitoken; further, by adding synonyms to the output, it provides extra source of information as the synonyms aka co-hyponyms are almost at the same level of taxonomic hierarchy, so they can easily be used to predict target word's hypernym.
The similar tripartite method is implemented in [1] for casual LM models (GPT-2 and LLaMA-2).

## Model description
The model consists of the three main blocks:
* initial generator (_word2vec_)
* filter (_dictionary_)
* filter + generator (_language model for mask-filling_)
  
### Generator: FastText
The most appropriate vector generator for this task is FastText [2], because it is designed to operate with subwords: using subword embeddings it can readily get representation for a previously unseen word. I trained the FastText model (FT) with the `gensim` library on 20% of texts from the UMBC Webbase corpus [3] used in SemEval-2018. The corpus is available at [CodaLab](https://competitions.codalab.org/competitions/17119#learn_the_details-terms_and_conditions).

The picture below shows the training process (comparing CBOW and Skip-Gram algorithms). The output was filtered with nltk's WordNet corpus, and the rest of candidates were evaluated. Model performance was monitored via MAP@15 metric, and it reached its peak at the 10th text. The resulting score turned out to be higher than that of the largest model pretrained by the fasttext's authors (see `crawl-300d-2M-subword` at https://fasttext.cc/docs/en/english-vectors.html). The pretrained model predictions are included in the full dataset file.

![alt text](https://github.com/feudor2/hypernym_masked_discoverer/blob/main/data/fasttext_training.png?raw=true)

The UMBC-trained version is available at https://disk.yandex.ru/d/dM3Vn2mlExzyZQ. The model predictions can also be found in one of the dataset files (see the following section for description).

### Filter: WordNet
The model uses nltk's WordNet `3.0` as a lexical database and for lemmatization. The taxonomical information is not extracted.

### Filter/Generator: transformer models for masked LM
BERT [4] architecture is convenient for filling gaps in patterns with the most probable words. BERT-like models are used in two ways. First, they can generate candidates by filling masks in prompts that share the structure with typical contextual patterns, ex. '<target> is a type of [MASK]', where we can expect hypernyms to appear in the place of `[MASK]`. Similarly the corresponding co-hyponym can be predicted from the co-hyponym pattern like '<target>, [MASK] and other of the same type'. Second, one can extract the words' probability to occur in the sentence from mask-filling models. So, if there are any predefined candidates, with the help of hypernym and co-hyponym prompts one can find more probable semantic relation, which can link them with the target word. In order to test this block, experiments were performed with two models: `bert-base-cased` and `FacebookAI/xlm-roberta-large` [5].

### Methods
Depending on the interaction between the model components, three methods were tested.
* Method 1 (_bert_hyper_): LM works only as generator and is applied to hypernym propmpts, no FT candidates used
* Method 2 (_ft+bert_hyper_): LM generation and FT candidates, that are the most probable hypernyms
* Method 3 (_ft+bert_cohypo_): LM generates hypernyms for FT candidates, that were predicted to be co-hyponyms
All these methods were repeated with iterative rearranging to make the top-level hypernyms obtain a higher rank. This procedure was suggested in [1].

## Data
The project uses SemEval-2018 Task 9 dataset [6]. It contains the `data` i.e. target terms to predict hypernyms for. The items are assigned to one of the two top-level (`base`) classes: `Entity` or `Concept`. 

![alt text](https://github.com/feudor2/hypernym_masked_discoverer/blob/main/data/concept-entity-ratio.png?raw=true)

For each term there is also a set of `gold` hypernyms, collected from semantic relation resources like WordNet and WikiData.
The dataset in `.tsv` format is stored in the `data` folder. The file `1A.dataset.tsv` contains the dataset proper, and `1A.all_ft_pretrained_pred.tsv` includes predicted nearest terms by two fasttext models (`crawl-300d-2M-subword` and the model mentioned above). Model predictions (`pred.wn` and `pred.trained` resp.) are filter with WordNet corpus.

## Repository contents
All necessary code files are stored in the `src` folder. Use the `main.py` file for testing the approach. Alternatively, you can launch the `main.ipynb` notebook from the `notebooks` folder in jupyter and run the same code cell by cell. The `src` folder contains the following modules:
* `for_masked_lm`: the model and some utilities to process the result
* `metrics`: a module used to calculate MAP and MRR metrics
* `prompt_utils`: this file includes the functions to fill the prompts and some examples used in the experiments
* `vector_model`: a module to load pretrained FT and get prediction from it
* `wordnet_helper`: an utility to profit from the nltk wordnet functionality
  
In addition, one can find three pickled objects required to run the `main.py`:
1. `dataset`: this is the minimal version of the dataset file
2. `fasttext_predictions`: the file contains a pandas.Series with predictions to fill the dataset; can be used without launching FastText
3. `stopwords` stores the list of unwanted words; this list can easily be modified in the code

All required packages are listed in `requirements.txt`, as well as the versions in which they were used. 

## Acknowledgements
The work is supported by Non-commercial Foundation for the Advancement of Science and Education INTELLECT, and my mentors, Natalia Loukachevitch and Alexander Ivchenko. 

## References
1. Tikhomirov, M., & Loukachevitch, N. (2024). Exploring Prompt-Based Methods for Zero-Shot Hypernym Prediction with Large Language Models. arXiv preprint arXiv:2401.04515.
2. Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2017). Enriching word vectors with subword information. Transactions of the association for computational linguistics, 5, 135-146.
3. Han, L., Kashyap, A. L., Finin, T., Mayfield, J., & Weese, J. (2013, June). UMBC_EBIQUITY-CORE: Semantic textual similarity systems. In Second joint conference on lexical and computational semantics (* SEM), volume 1: Proceedings of the main conference and the shared task: Semantic textual similarity (pp. 44-52).
4. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.
5. Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., ... & Stoyanov, V. (2019). Unsupervised cross-lingual representation learning at scale. arXiv preprint arXiv:1911.02116.
6. Camacho-Collados, J., Bovi, C. D., Anke, L. E., Oramas, S., Pasini, T., Santus, E., ... & Saggion, H. (2018, June). SemEval-2018 task 9: Hypernym discovery. In Proceedings of the 12th international workshop on semantic evaluation (pp. 712-724).
