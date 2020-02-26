# -*- coding: utf-8 -*-
"""
Сохраняем модель w2v  в формате KeyedVectors для оптимизации времени загрузки
Результат сохраняем в tmp.
"""

import os
import gensim

tmp_dir = '../tmp'

w2v_path = os.path.expanduser('~/polygon/w2v/w2v.CBOW=1_WIN=5_DIM=64.bin')

print(u'Loading w2v from {}'.format(w2v_path))
w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
w2v_dims = len(w2v.vectors[0])

# Сохраняем в формате для быстрой загрузки, будет 2 файла.
w2v.save(os.path.join(tmp_dir, 'w2v.kv'))
