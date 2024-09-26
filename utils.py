from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from dataclasses import dataclass, field
from typing import Optional, List, Any, Union, Dict, Tuple
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler
import pickle
import os
from itertools import chain
import argparse

@dataclass
class DataCollatorForDecoupledTraining:
    tokenizer: PreTrainedTokenizerBase
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:

        input_ids = [e['input_ids'] for e in examples]
        attention_mask = [e['attention_mask'] for e in examples]
        special_tokens_mask = [e['special_tokens_mask'] for e in examples]
        token_lang_ids = [[e['token_lang_ids']] * len(e['input_ids']) for e in examples]
        token_script_ids = [[e['token_script_ids']] * len(e['input_ids']) for e in examples]
        
        input_ids = self._tensorize_batch(input_ids, padding_value=self.tokenizer.pad_token_id)
        attention_mask = self._tensorize_batch(attention_mask, padding_value=0)
        special_tokens_mask = self._tensorize_batch(special_tokens_mask, padding_value=1)
        token_lang_ids = self._tensorize_batch(token_lang_ids, padding_value=0)
        token_script_ids = self._tensorize_batch(token_script_ids, padding_value=0)

        batch = dict()

        batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
            input_ids, special_tokens_mask=special_tokens_mask
        )

        batch['attention_mask'] = attention_mask
        # batch['special_tokens_mask'] = special_tokens_mask
        batch['token_lang_ids'] = token_lang_ids
        batch['token_script_ids'] = token_script_ids

        return batch

    def _tensorize_batch(self, examples: List[Union[torch.Tensor, np.ndarray]], padding_value) -> torch.Tensor:
        examples = [torch.tensor(ex) if not isinstance(ex, torch.Tensor) else ex for ex in examples]
        return torch.stack(examples, dim=0)

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


# load from each language_script file (text file)
def load_decoupled_training_dataset(file_path, tokenizer, max_seq_length, pad_to_multiple_of_8,
                                    language_dict, script_dict,
                                    model_args, data_args, training_args):

    filenames = os.listdir(file_path)
    filenames = [file_path + '/' + filename for filename in filenames]
    data_files = {filename.split('/')[-1].split('.')[0]: filename for filename in filenames}
    datasets = load_dataset(
        'text',
        cache_dir=model_args.cache_dir,
        data_files=data_files
    )

    def preprocess_function(examples):
        tokenized_text = tokenizer(examples["text"], return_special_tokens_mask=True)
        model_inputs = dict()
        model_inputs['input_ids'] = tokenized_text['input_ids']
        model_inputs['attention_mask'] = tokenized_text['attention_mask']
        model_inputs['special_tokens_mask'] = tokenized_text['special_tokens_mask']
        return model_inputs

    with training_args.main_process_first(desc="dataset map tokenization"):
        for key in datasets.keys():
            datasets[key] = datasets[key].map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=['text'],
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on every text in dataset",
            )

    def group_texts(examples, language, script):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // max_seq_length) * max_seq_length
        result = {
            k: [t[i: i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        result['token_lang_ids'] = [language_dict[language] for _ in range(len(result['input_ids']))]
        result['token_script_ids'] = [script_dict[script] for _ in range(len(result['input_ids']))]
        return result

    # concatenate for each langauge_script
    with training_args.main_process_first(desc="grouping texts together"):
        for key in datasets.keys():
            language, script = key.split('_')
            datasets[key] = datasets[key].map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {max_seq_length}",
                fn_kwargs={'language': language, 'script': script},
            )

    # concatenate all datasets to one single dataset
    final_dataset = concatenate_datasets(list(datasets[key] for key in datasets.keys()))
    data_collator = DataCollatorForDecoupledTraining(
        tokenizer=tokenizer,
        mlm_probability=data_args.mlm_probability,
        pad_to_multiple_of=8 if pad_to_multiple_of_8 else None
    )

    return final_dataset, data_collator
