# -*- coding: utf-8 -*-


from __future__ import division  # for python2 compatibility
from __future__ import print_function

import os
import io
import re
import operator
import numpy as np


def is_num(token):
    return re.match('^[0-9]+$', token)


word2features = dict()


def get_word_features(word, word2tags, w2v):
    if word in word2features:
        return word2features[word]

    lword = word.lower().replace('ё', 'е')

    features = set()

    if word in ('<beg>', '<beg>'):
        features.add((word, 1.0))
    elif is_num(word):
        features.add(('<number>', 1.0))
    elif word[0] in u'‼≠™®•·[¡+<>`~;.,‚?!-…№”“„{}|‹›/\'"–—_:«»*]()‘’≈':
        features.add((u'punct_{}'.format(ord(word[0])), 1.0))
    elif word[0] in u'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
        features.add((u'<latin>', 1.0))
    else:
        for tagset in word2tags[lword]:
            for tag in tagset.split(' '):
                features.add((tag, 1.0))

    if word[0] in u'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯABCDEFGHIJKLMNOPQRSTUVWXYZ':
        features.add((u'<upper>', 1.0))

    if lword in w2v:
        v = w2v[lword]
        for ix, x in enumerate(v):
            features.add((u'w2v[{}]'.format(ix), x))

    features = list(features)
    word2features[word] = features
    return features
