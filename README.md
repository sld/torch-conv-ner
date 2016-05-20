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

# Эксперименты

Все эксперименты проводились на AWS g2.2xlarge с использование GPU.

1.
  Запустить скрипт `bash experiments/convolution-net.sh`. По прошествию примерно 5 часов
  обучится модель с F1 в районе 87.5%.
  В папку snapshots сохраняется модель с лучшей F1 мерой каждые 2 эпохи.
  В ней же можно посмотреть логи обучения.

# Ссылки

- [Natural Language Processing (almost) from Scratch](https://arxiv.org/abs/1103.0398)
- http://ml.nec-labs.com/senna/
- https://github.com/patverga/torch-ner-nlp-from-scratch
- https://github.com/attardi/deepnl
