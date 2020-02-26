# -*- coding: utf-8 -*-

from __future__ import division  # for python2 compatibility
from __future__ import print_function

import os
import numpy as np
import json

import keras
from keras.models import load_model

from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy

from word2features import get_word_features


PAD_WORD = u''
PAD_FEATURE = u'<padding>'
NULL_LABEL = u'<padding>'


class Tagger(object):
    def __init__(self, word2tags, w2v):
        self.word2tags = word2tags
        self.w2v = w2v
        self.nb_features = -1
        self.max_text_len = -1
        self.feature2index = None
        self.index2label = None
        self.X_data = None

    def load(self, model_dir):
        with open(os.path.join(model_dir, 'rupostagger2.config'), 'r') as f:
            config = json.load(f)
        self.nb_features = config['nb_features']
        self.max_text_len = config['max_text_len']
        self.feature2index = config['feature2index']
        self.index2label = dict((i, l) for (l, i) in config['label2index'].items())
        self.X_data = np.zeros((1, self.max_text_len, self.nb_features), dtype=np.float32)

        custom_objects = {'CRF': CRF,
                          'crf_loss': crf_loss,
                          'crf_viterbi_accuracy': crf_viterbi_accuracy}
        self.model = load_model(os.path.join(model_dir, 'rupostagger2.model'), custom_objects=custom_objects)

    def tag(self, words0):
        """
        Метод выполняет разметку списка слов метками. Каждая метка - набор тегов, разделенных символов "|"
        :param words: список слов - юникодных строк.
        :return: список пар (слово, метка)
        """
        #words = ['<beg>'] + words0 + ['<end>']
        words = words0
        nwords = len(words)
        if nwords > self.max_text_len:
            raise RuntimeError('Maximum length of text {} exceeded'.format(self.max_text_len))

        nwords0 = len(words0)

        self.X_data.fill(0)
        for iword, word in enumerate(words):
            features = get_word_features(word, self.word2tags, self.w2v)
            for feature_name, feature_val in features:
                if feature_name in self.feature2index:
                    self.X_data[0, iword, self.feature2index[feature_name]] = feature_val

        # заполнители справа
        npad = max(0, self.max_text_len - nwords)
        for ipad in range(npad):
            self.X_data[0, nwords+ipad, self.feature2index[PAD_FEATURE]] = 1.0

        # получаем от модели коды меток
        y_pred = self.model.predict(self.X_data, verbose=0)[0]
        y_pred = np.argmax(y_pred, axis=-1)
        #y_pred = y_pred[1:nwords0+1]

        # Результат - список слов и предсказанных меток
        return [(word, self.index2label[y]) for (word, y) in zip(words0, y_pred)]
