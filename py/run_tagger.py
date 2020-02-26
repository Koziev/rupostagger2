# -*- coding: utf-8 -*-

from __future__ import division  # for python2 compatibility
from __future__ import print_function

import os
import io
import gc
import json
import numpy as np

import gensim
import ruword2tags

from tagger import Tagger


if __name__ == '__main__':
    model_dir = '../tmp'

    # Эмбеддинги слов могут быть нужны, если модель тренировалась с ними.
    #w2v_path = os.path.expanduser('~/polygon/chatbot/tmp/w2v.kv')
    w2v_path = os.path.join(model_dir, 'w2v.kv')
    wordchar2vector_path = '~/polygon/chatbot/data/wordchar2vector.dat'

    word2tags = ruword2tags.RuWord2Tags()
    word2tags.load()

    # Загрузим конфиг модели
    with open(os.path.join(model_dir, 'rupostagger2.config'), 'r') as f:
        config = json.load(f)

    w2v = None
    if config['use_w2v']:
        if config['use_wc2v']:
            print(u'Loading the wordchar2vector model from "{}"'.format(wordchar2vector_path))
            wc2v = gensim.models.KeyedVectors.load_word2vec_format(wordchar2vector_path, binary=False)
            wc2v_dims = len(wc2v.vectors[0])

            print('Loading w2v from "{}"...'.format(w2v_path))
            #w2v = FastText.load_fasttext_format(w2v_path)
            w2v = gensim.models.KeyedVectors.load(w2v_path, mmap='r')
            w2v_dims = len(w2v.vectors[0])

            vocabul_path = os.path.join(model_dir, 'rupostagger2.vocabulary.dat')
            with io.open(vocabul_path, 'r', encoding='utf-8') as rdr:
                all_words = [s.strip() for s in rdr]

            word_dims = w2v_dims + wc2v_dims
            word2vec = dict()
            for word in all_words:
                v = np.zeros(word_dims)

                if word in wc2v:
                    v[w2v_dims:] = wc2v[word]

                if word in w2v:
                    v[:w2v_dims] = w2v[word]

                word2vec[word] = v

            del w2v
            del wc2v
            gc.collect()

            w2v = word2vec
        else:
            print('Loading w2v from "{}"...'.format(w2v_path))
            #w2v = FastText.load_fasttext_format(w2v_path)
            w2v = gensim.models.KeyedVectors.load(w2v_path, mmap='r')
    else:
        w2v = dict()

    tagger = Tagger(word2tags, w2v)

    # Модель была ранее обучена (см. run_trainer.py) и файлы данных сохранены в указанный каталог.
    tagger.load(model_dir=model_dir)

    while True:
        sent = input(':>')
        labels = tagger.tag(sent.split())
        for word, label in labels:
            print(u'{} --> {}'.format(word, label))
