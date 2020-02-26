# -*- coding: utf-8 -*-


from __future__ import division  # for python2 compatibility
from __future__ import print_function

import os
import io
import gc
import operator
import numpy as np
import json

import logging

import keras
from keras.models import save_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input
from keras.layers import recurrent
from keras.layers.wrappers import Bidirectional
from keras.models import Model

from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy

from sklearn.model_selection import train_test_split

from word2features import get_word_features


PAD_WORD = u''
PAD_FEATURE = u'<padding>'
NULL_LABEL = u'<padding>'


class Trainer(object):
    def __init__(self):
        pass

    def train(self, corpus_path, word2tags, w2v, out_folder, params2, max_sent_len=50, max_nb_samples=10000):
        logging.info(u'Analysis of corpus "{}"'.format(corpus_path))
        all_features = set([PAD_FEATURE])
        all_labels = set([NULL_LABEL])
        max_text_len = 0
        nb_samples = 0
        samples = []
        with io.open(corpus_path, 'r', encoding='utf-8') as rdr:
            tokens = []
            for line in rdr:
                line = line.strip()
                if len(line) == 0:
                    if len(tokens) <= max_sent_len:
                        max_text_len = max(max_text_len, len(tokens))
                        nb_samples += 1
                        samples.append(tokens)
                    tokens = []
                else:
                    tx = line.split('\t')
                    word = tx[1]
                    label = tx[2]
                    all_labels.add(label)

                    fx = get_word_features(word, word2tags, w2v)
                    all_features.update(map(operator.itemgetter(0), fx))
                    tokens.append((word, label))

        nb_labels = len(all_labels)
        nb_features = len(all_features)
        logging.info('nb_labels=%d', nb_labels)
        logging.info('nb_features=%d', nb_features)
        logging.info('max_text_len=%d', max_text_len)

        label2index = dict((l, i) for (i, l) in enumerate(all_labels))
        feature2index = dict((f, i) for (i, f) in enumerate(all_features))

        index2class = dict((i, l.split('|')[0]) for (l, i) in label2index.items())

        config = {'nb_labels': nb_labels,
                  'nb_features': nb_features,
                  'max_text_len': max_text_len,
                  'label2index': label2index,
                  'feature2index': feature2index}
        config.update(params2)
        with open(os.path.join(out_folder, 'rupostagger2.config'), 'w') as f:
            json.dump(config, f, indent=4)

        gc.collect()

        if len(samples) > max_nb_samples:
            samples = sorted(samples, key=lambda z: -len(z))[:max_nb_samples]

        nb_samples = len(samples)
        X_data = np.zeros((nb_samples, max_text_len, nb_features), dtype=np.float32)
        y_data = np.zeros((nb_samples, max_text_len, nb_labels), dtype=np.bool)
        nbtokens_data = np.zeros(nb_samples, dtype=np.int32)  # число токенов в каждом сэмпле

        logging.info('Vectorization of %d samples', nb_samples)

        for isample, tokens in enumerate(samples):
            nbtokens_data[isample] = len(tokens)
            for iword, (word, label) in enumerate(tokens):
                features = get_word_features(word, word2tags, w2v)
                for feature_name, feature_val in features:
                    X_data[isample, iword, feature2index[feature_name]] = feature_val
                y_data[isample, iword, label2index[label]] = 1

            ntokens = len(tokens)
            npad = max(0, max_text_len - ntokens)
            for ipad in range(npad):
                iword = ntokens + ipad
                X_data[isample, iword, feature2index[PAD_FEATURE]] = 1.0
                y_data[isample, iword, label2index[NULL_LABEL]] = 1
            isample += 1

        del samples
        samples = None
        gc.collect()

        X_train, X_val,\
        y_train, y_val,\
        nbtokens_train, nbtokens_val = train_test_split(X_data, y_data, nbtokens_data, test_size=0.1)

        input = Input(shape=(max_text_len, nb_features,), dtype='float32', name='input')
        net = input

        for _ in range(1):
            net = Bidirectional(recurrent.LSTM(units=nb_features*2,
                                               dropout=0.0,
                                               return_sequences=True))(net)

        net = CRF(units=nb_labels, sparse_target=False)(net)
        model = Model(inputs=[input], outputs=net)
        model.compile(loss=crf_loss, optimizer='nadam', metrics=[crf_viterbi_accuracy])
        model.summary()

        weights_path = os.path.join(out_folder, 'rupostagger2.weights')
        monitor_metric = 'val_crf_viterbi_accuracy'

        model_checkpoint = ModelCheckpoint(weights_path, monitor=monitor_metric,
                                           verbose=1, save_best_only=True, mode='auto')
        early_stopping = EarlyStopping(monitor=monitor_metric, patience=10, verbose=1, mode='auto')

        callbacks = [model_checkpoint, early_stopping]

        model.fit(x=X_train, y=y_train,
                  validation_data=(X_val, y_val),
                  epochs=100,
                  batch_size=150,
                  verbose=2,
                  callbacks=callbacks)

        model.load_weights(weights_path)

        save_model(model, os.path.join(out_folder, 'rupostagger2.model'))
        os.remove(weights_path)


        nb_total = 0
        nb_good = 0

        nb_val_NOUN = 0  # всего меток NOUN в валидационных данных
        nb_pred_NOUN = 0  # всего меток NOUN в предсказаниях, включая неверные
        nb_pred1_NOUN = 0  # сколько меток NOUN правильно предсказано, для вычисления полноты

        nb_val_ADJ = 0
        nb_pred_ADJ = 0
        nb_pred1_ADJ = 0

        nb_val_VERB = 0
        nb_pred_VERB = 0
        nb_pred1_VERB = 0

        batch_size = 1000
        nb_batch = y_val.shape[0] // batch_size
        for ibatch in range(nb_batch):
            i0 = ibatch * batch_size
            i1 = i0 + batch_size

            y_pred = model.predict(X_val[i0:i1])

            # оценка точности per instance
            nb_total += batch_size
            nb_good += sum(np.array_equal(y_pred[i, :], y_val[i0+i, :]) for i in range(batch_size))

            # оценка точности per token
            yy_pred = np.argmax(y_pred[i0: i1], axis=-1)
            yy_val = np.argmax(y_val, axis=-1)

            for irow in range(yy_pred.shape[0]):
                # Чтобы не учитывать метки на заполнителях, урезаем каждый сэмпл до его фактической длины
                y1 = yy_pred[irow][:nbtokens_val[irow]]
                y2 = yy_val[irow][:nbtokens_val[irow]]
                n1 = np.sum(np.equal(y1, y2))
                nb_good += n1
                nb_total += len(y1)

                for yy1, yy2 in zip(y1, y2):
                    pred_class = index2class[yy1]
                    val_class = index2class[yy2]

                    if val_class == 'NOUN':
                        nb_val_NOUN += 1
                    elif val_class == 'ADJ':
                        nb_val_ADJ += 1
                    elif val_class == 'VERB':
                        nb_val_VERB += 1

                    if pred_class == 'NOUN':
                        nb_pred_NOUN += 1
                    elif pred_class == 'ADJ':
                        nb_pred_ADJ += 1
                    elif pred_class == 'VERB':
                        nb_pred_VERB += 1

                    if pred_class == 'NOUN' and val_class == 'NOUN':
                        nb_pred1_NOUN += 1

                    if pred_class == 'ADJ' and val_class == 'ADJ':
                        nb_pred1_ADJ += 1

                    if pred_class == 'VERB' and val_class == 'VERB':
                        nb_pred1_VERB += 1

        acc_perinstance = nb_good / float(nb_total)
        logging.info('acc_perinstance=%5.3f', acc_perinstance)

        acc_pertoken = nb_good / float(nb_total)
        logging.info('acc_pertoken=%5.3f', acc_pertoken)

        # Полнота для NOUN - сколько NOUN из всех ожидавшихся модель смогла предсказать
        recall_NOUN = nb_pred1_NOUN / float(nb_val_NOUN)

        # Точность для NOUN - какая доля предсказаний NOUN корректна
        precision_NOUN = nb_pred1_NOUN / float(nb_pred_NOUN)

        f1_NOUN = 2.0 * precision_NOUN * recall_NOUN / (precision_NOUN + recall_NOUN)
        logging.info('recall_NOUN={:5.3f} precision_NOUN={:5.3f} f1_NOUN={:5.3f}'.format(recall_NOUN, precision_NOUN, f1_NOUN))

        recall_ADJ = nb_pred1_ADJ / float(nb_val_ADJ)
        precision_ADJ = nb_pred1_ADJ / float(nb_pred_ADJ)
        f1_ADJ = 2.0 * precision_ADJ * recall_ADJ / (precision_ADJ + recall_ADJ)
        logging.info('recall_ADJ={:5.3f}  precision_ADJ={:5.3f}  f1_ADJ={:5.3f}'.format(recall_ADJ, precision_ADJ, f1_ADJ))

        recall_VERB = nb_pred1_VERB / float(nb_val_VERB)
        precision_VERB = nb_pred1_VERB / float(nb_pred_VERB)
        f1_VERB = 2.0 * precision_VERB * recall_VERB / (precision_VERB + recall_VERB)
        logging.info('recall_VERB={:5.3f} precision_VERB={:5.3f} f1_VERB={:5.3f}'.format(recall_VERB, precision_VERB, f1_VERB))

        return acc_pertoken, acc_perinstance
