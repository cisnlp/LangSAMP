# LangSAMP

This is the repository for **LangSAMP** (Language-Script Aware Multilingual Pretraining), a method that incorporates both **language** and **script** embeddings to enhance representation learning while maintaining a simple architecture. Specifically, we integrate these embeddings into the output of the transformer blocks before passing the final representations to the language modeling head for prediction. Our pretraining ease the burden of encoding language-specific information in token embeddings, thus improving their language neutrality, which is beneficial for crosslingual transfer.

<div style="text-align: center;">
    <img src="/figures/pipeline.png" width="800" height="400" />
</div>

Paper on arXiv: https://arxiv.org/abs/2409.18199


## Continued Pretraining

We use the [Glot500-c](https://github.com/cisnlp/Glot500) corpus for continued-pretraining our models. The dataset contains more than 500 languages.

To continued-pretrain the model, run:

```
bash run_mlm.sh
```

You can change the .sh files for specifying ```--use_lang_embedding``` for using language embeddings or ```--use_script_embedding``` for using script embeddings. ```full_dicts.pkl``` is a pickle file contains the dictionaries mapping language codes ISO-639-3 to integers (starting from 0) and script codes to integers (starting from 0), which are required to initialize the language embeddings and script embeddings.


## Evaluation

### Dataset Preparation

Please refer to [Glot500](https://github.com/cisnlp/Glot500) for downloading the datasets used for evaluation.

### Sentence Retrieval - Bible

For SR-B, first go to ``evaluation/retrieval`` and run:

```
bash evaluate_retrieval_bible_all.sh
```


### Sentence Retrieval - Tatoeba

For SR-T, first go to ``evaluation/retrieval`` and run:

```
bash evaluate_retrieval_tatoeba_all.sh
```

### Text Classification - Taxi1500

First go to ``evaluation/taxi1500`` and run:

```
bash evaluate_all.sh
```

### Text Classification - SIB200

First go to ``evaluation/sib200`` and run:

```
bash evaluate_sib_all.sh
```

### Named Entity Recognition

For NER, first go to ``evaluation/tagging`` and run:
```
bash evaluate_all_ner.sh
```

### Part-Of-Speech Tagging

For POS, first go to ``evaluation/tagging`` and run:
```
bash evaluate_all_pos.sh
```

## Citation

If you find our model, data or the overview of data useful for your research, please cite:

```
@misc{liu2024langsamplanguagescriptawaremultilingual,
      title={LangSAMP: Language-Script Aware Multilingual Pretraining}, 
      author={Yihong Liu and Haotian Ye and Chunlan Ma and Mingyang Wang and Hinrich Schütze},
      year={2024},
      eprint={2409.18199},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.18199}, 
}
```

## Acknowledgements

This repository is built on top of [transformers](https://github.com/huggingface/transformers), [xtreme](https://github.com/google-research/xtreme), [Glot500](https://github.com/cisnlp/Glot500).
