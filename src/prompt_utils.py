import re

from wordnet_helper import WordNetHelper

def prepare_prompts(word, prompts, cohypo="", article=True):
    """Prepare prompts for 

        Parameters
        ----------
        prompt : str
            a prompt with a place for the target (use `♥`) and optionally for an extra co-hyponym (use `♠` symbol)
        word : str
            a target word (to replace `♥`)
        cohypo : str, optional
            an additional co-hyponym to use in mixed prompts (to replace `♠`)
        article : bool, optional
            if True, article will be added before the target depending on its grammatical form
            
        Returns
        -------
        list
            list of prompts to be used by the HypernymMaskedModel
    """
    def replace_tokens(word, prompt, cohypo=cohypo):
        def article_replace(token, mask, prompt):
            try:
                if token[0].isupper() or WN.lemmatize(token) != token:
                    return re.sub(mask, token, prompt)
                if token[0] in "aeoiu":
                    return re.sub(mask, "an " + token, prompt)
                return re.sub(mask, "a " + token, prompt)
            except IndexError:
                return re.sub(mask, token, prompt)
        
        WN = WordNetHelper()
        # no article
        prompt1 = re.sub("♥", word, prompt)
        prompt1 = re.sub("♠", cohypo, prompt1)
        # with article:
        if article:
            prompt2 = article_replace(word, "♥", prompt)
            prompt2 = article_replace(cohypo, "♠", prompt2)
            return prompt1, prompt2
        return prompt1

    output = []
    for p in prompts:
        if article:
            output.extend(replace_tokens(word, prompt=p, cohypo=cohypo))
        else:
            output.append(replace_tokens(word, prompt=p, cohypo=cohypo))
    return output

def normalize_prompts(prompts):
    '''Function to replace special characters by some more clear units'''
    prompts = [re.sub('♥', '<target>', prompt) for prompt in prompts]
    prompts = [re.sub('♠', '<cohypo>', prompt) for prompt in prompts]
    prompts = [re.sub('_', '[MASK]', prompt) for prompt in prompts]
    return prompts

basic_prompts = {
    'cohypo_prompts': [
        'such as ♥, _ and other of the same type.',
        'such as ♥ and _ of the same type.',
        'such as ♥ or _.',
    ],
    'best_cohypo_prompt': ['such as ♥ and _ of the same type.'],
    'hyper_prompts': [
        '♥ is a _.',
        '_, such as ♥.',
        'a _, such as ♥.',
        '♥ is a type of _.',
        'My favorite _ is ♥.',
        ## MT&NL'24 prompts
        '♥ or some other _.',
        '♥ or any other _.',
        '♥ and any other _.',
    ],
    'mixed_prompts': [
        ## hyper with cohypo
        'My favorite _ is either ♥ or ♠.',
        '♥ is a _. So is ♠.',
    ]
}