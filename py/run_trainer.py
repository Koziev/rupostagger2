# -*- coding: utf-8 -*-
"""
Пример запуска тренировки нейросетевого теггера.
Используется готовый корпус - файл ../data/samples.dat, который генерируется
скриптом prepare_dataset.py в rupostagger
"""

from __future__ import division  # for python2 compatibility
from __future__ import print_function

import os
import logging
import io
import gc
import logging.handlers
import absl.logging  # https://github.com/tensorflow/tensorflow/issues/26691

import numpy as np
import gensim
from gensim.models.wrappers import FastText
import ruword2tags

from trainer import Trainer


use_w2v = True
use_wc2v = True


if __name__ == '__main__':
    corpus_path = '../data/samples.dat'
    tmp_dir = '../tmp'

    # настраиваем логирование в файл и эхо-печать в консоль
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')

    logfile_path = os.path.join(tmp_dir, 'rupostagger2.trainer.log')
    lf = logging.FileHandler(logfile_path, mode='w')
    lf.setLevel(logging.INFO)
    lf.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logging.getLogger('').addHandler(lf)

    logging.info('STARTED')

    trainer = Trainer()

    logging.info('Loading dictionary...')
    word2tags = ruword2tags.RuWord2Tags()
    word2tags.load()

    w2v = None
    if use_w2v:
        w2v_path = os.path.join(tmp_dir, 'w2v.kv')
        #w2v_path = os.path.join('/home/inkoziev/polygon/w2v/fasttext.CBOW=1_WIN=5_DIM=64')
        wordchar2vector_path = '~/polygon/chatbot/data/wordchar2vector.dat'

        if 'fasttext' in w2v_path:
            w2v = FastText.load_fasttext_format(w2v_path)
        else:
            if use_wc2v:
                logging.info(u'Loading the wordchar2vector model from "%s"', wordchar2vector_path)
                wc2v = gensim.models.KeyedVectors.load_word2vec_format(wordchar2vector_path, binary=False)
                wc2v_dims = len(wc2v.vectors[0])
                logging.info('wc2v_dims=%d', wc2v_dims)

                logging.info('Loading w2v from "%s"...', w2v_path)
                #w2v = FastText.load_fasttext_format(w2v_path)
                w2v = gensim.models.KeyedVectors.load(w2v_path, mmap='r')
                w2v_dims = len(w2v.vectors[0])
                logging.info('w2v_dims=%d', w2v_dims)

                # Соберем лексикон
                all_words = set()
                with io.open(corpus_path, 'r', encoding='utf-8') as rdr:
                    for iline, line in enumerate(rdr):
                        line = line.strip()
                        if line:
                            tx = line.split('\t')
                            if len(tx) == 3:
                                all_words.add(tx[1].lower())

                vocabul_path = os.path.join(tmp_dir, 'rupostagger2.vocabulary.dat')
                with io.open(vocabul_path, 'w', encoding='utf-8') as wrt:
                    for word in sorted(all_words):
                        wrt.write('{}\n'.format(word))
                logging.info('Vocabulary: %d words stored in "%s"', len(all_words), vocabul_path)

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
                logging.info('Loading w2v from "%s"...'.format(w2v_path))
                #w2v = FastText.load_fasttext_format(w2v_path)
                w2v = gensim.models.KeyedVectors.load(w2v_path, mmap='r')
    else:
        w2v = dict()

    logging.info('Training...')

    params2 = {'use_w2v': use_w2v, 'use_wc2v': use_wc2v}
    trainer.train(corpus_path, word2tags, w2v, tmp_dir, max_sent_len=30, max_nb_samples=500000, params2=params2)
    logging.info('All done.')
