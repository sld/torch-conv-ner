# Environment

* Python 2.7.11
* [torch](http://torch.ch/docs/getting-started.html)
* [CoNLL-2003](http://www.cnts.ua.ac.be/conll2003/ner/) corpora

# Preparation

1. Put CoNLL-2003 corpora into data/conll2003 folder.
Your CoNLL 2003 files should be renamed and placed as below:
  - data/conll2003/eng.testa.dev;
  - data/conll2003/eng.testb.test;
  - data/conll2003/eng.train.
2. Run script `bash utils/prepare-senna-data.sh`. It downloads [senna embeddings](http://ml.nec-labs.com/senna/download.html),
gazeteers and puts them into data/.
3. Install [torch](http://torch.ch/docs/getting-started.html) and python libs from requirements.txt (`pip install -r requirements.txt`).

# Experiments

All experiments are done by using AWS g2.2xlarge with GPU.

1.
  Run `bash experiments/convolution-net.sh`. After about 5 hours
  model with 87.5% F1 will be learnt.
  In snapshots directory will be saved the model with the best F1 (each 2 epochs).
  Learning logs also saving there.
  
  For example, learning log for 74 epoch:
  ```
  processed 46666 tokens with 5648 phrases; found: 5778 phrases; correct: 4990.
  accuracy:  97.65%; precision:  86.36%; recall:  88.35%; FB1:  87.34
                LOC: precision:  90.35%; recall:  90.89%; FB1:  90.62  1678
               MISC: precision:  73.82%; recall:  75.50%; FB1:  74.65  718
                ORG: precision:  80.69%; recall:  85.01%; FB1:  82.79  1750
                PER: precision:  93.87%; recall:  94.74%; FB1:  94.31  1632
  ```

# References

- [Named entity recognition using syntactic and semantic features and neural networks](http://www.dialog-21.ru/media/3475/yusupov.pdf)
- [Natural Language Processing (almost) from Scratch](https://arxiv.org/abs/1103.0398)
- http://ml.nec-labs.com/senna/
- https://github.com/patverga/torch-ner-nlp-from-scratch
- https://github.com/attardi/deepnl
