# Окружение

* Python 2.7.11
* [torch](http://torch.ch/docs/getting-started.html)
* Корпус [CoNLL-2003](http://www.cnts.ua.ac.be/conll2003/ner/)

# Подготовка

1. Положить корпуса CoNLL-2003 в папку data/conll2003.
В результате должно быть так:
  - data/conll2003/eng.testa.dev;
  - data/conll2003/eng.testb.test;
  - data/conll2003/eng.train.
2. Запустить скрипт `bash utils/prepare-senna-data.sh`. Он скачивает [senna embeddings](http://ml.nec-labs.com/senna/download.html),
газетиры и кладет их в папку data/.
3. Установить torch и зависимости для Питона из файла requirements.txt (`pip install -r requirements.txt`).

# Эксперименты

Все эксперименты проводились на AWS g2.2xlarge с использование GPU.

1.
  Запустить скрипт `bash experiments/convolution-net.sh`. По прошествию примерно 5 часов
  обучится модель с F1 в районе 87.5%.
  В папку snapshots сохраняется модель с лучшей F1 мерой каждые 2 эпохи.
  В ней же можно посмотреть логи обучения.
  Результаты к 74 эпохе на тестовой выборке:
  ```
  processed 46666 tokens with 5648 phrases; found: 5778 phrases; correct: 4990.
  accuracy:  97.65%; precision:  86.36%; recall:  88.35%; FB1:  87.34
                LOC: precision:  90.35%; recall:  90.89%; FB1:  90.62  1678
               MISC: precision:  73.82%; recall:  75.50%; FB1:  74.65  718
                ORG: precision:  80.69%; recall:  85.01%; FB1:  82.79  1750
                PER: precision:  93.87%; recall:  94.74%; FB1:  94.31  1632
  ```

# Ссылки

- [Natural Language Processing (almost) from Scratch](https://arxiv.org/abs/1103.0398)
- http://ml.nec-labs.com/senna/
- https://github.com/patverga/torch-ner-nlp-from-scratch
- https://github.com/attardi/deepnl
