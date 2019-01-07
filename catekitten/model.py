from catekitten.data import Loader
from catekitten.data import get_category_map
from catekitten.transform import PosTokenizer
from catekitten.utils import lowercase, gen_batch
from catekitten.factory import NETWORKS
from catekitten.hierarchy import child_category, parent_category
from catekitten import param

import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import string
import os
import re

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras_tqdm import TQDMNotebookCallback 
from joblib import Parallel, delayed


DATA_DIR = "./data/prep/textonly.h5"
LABEL_DIR = "./data/raw/cate1.json"


def load_datasets(data_dir=DATA_DIR):
    """Load data 
    
    Args:
        data_dir (str): directory of converted h5 data
    
    Returns:
        Loader: hdf data
    """

    train = Loader(data_dir, subset_name='train')
    dev = Loader(data_dir, subset_name='dev')
    test = Loader(data_dir, subset_name='test')
    return train, dev, test


def load_category_labels(data_dir=LABEL_DIR):
    return get_category_map(data_dir)


class TextFeatureClassifier(object):

    def __init__(self):
        # set random seed for reproducibility
        np.random.seed(39)
        tf.set_random_seed(39)

        self.train_dataset, self.dev_dataset, self.test_dataset = \
            load_datasets()
        self.category_labels = load_category_labels()
        self.columns = self.train_dataset.columns
        
        self.text_feature_columns = ['product', 'brand', 'maker', 'model']
        self.phase = ['train', 'dev', 'test']
        self.categories = ['b','m','s','d']

        # replace unknown category to 0 
        self.train_y = {
            'b': self.train_dataset['bcateid'].replace(to_replace=-1, value=0),
            'm': self.train_dataset['mcateid'].replace(to_replace=-1, value=0),
            's': self.train_dataset['scateid'].replace(to_replace=-1, value=0),
            'd': self.train_dataset['dcateid'].replace(to_replace=-1, value=0),
        }

        self.dev_y = {
            'b': self.dev_dataset['bcateid'].replace(to_replace=-1, value=0),
            'm': self.dev_dataset['mcateid'].replace(to_replace=-1, value=0),
            's': self.dev_dataset['scateid'].replace(to_replace=-1, value=0),
            'd': self.dev_dataset['dcateid'].replace(to_replace=-1, value=0),
        }

        self.test_y = {
            'b': self.test_dataset['bcateid'].replace(to_replace=-1, value=0),
            'm': self.test_dataset['mcateid'].replace(to_replace=-1, value=0),
            's': self.test_dataset['scateid'].replace(to_replace=-1, value=0),
            'd': self.test_dataset['dcateid'].replace(to_replace=-1, value=0),
        }

        self.y = {
            'train': self.train_y,
            'dev': self.dev_y,
            'test': self.test_y
        }

        self.preds = {
            'b': None,
            'm': None,
            's': None,
            'd': None,
        }

        self.encoders = None
        
    def initialize_features(self):
        # Initialize text feature maps 
        self.train_text_feature_map = {
            'product': self.train_dataset['product'],
            'brand': self.train_dataset['brand'],
            'maker': self.train_dataset['maker'],
            'model': self.train_dataset['model'],
        }

        self.dev_text_feature_map = {
            'product': self.dev_dataset['product'],
            'brand': self.dev_dataset['brand'],
            'maker': self.dev_dataset['maker'],
            'model': self.dev_dataset['model'],
        }

        self.test_text_feature_map = {
            'product': self.test_dataset['product'],
            'brand': self.test_dataset['brand'],
            'maker': self.test_dataset['maker'],
            'model': self.test_dataset['model'],
        }
        
        self.text_feature_maps  = {
            'train': self.train_text_feature_map,
            'dev': self.dev_text_feature_map,
            'test': self.test_text_feature_map
        }

        self.x = {
            'train': np.array([]),
            'dev': np.array([]),
            'test': np.array([])
        }

    def describe_labels(self):
        for letter in ['b','m','s','d']:
            print("number of %scateid classes:" % letter, 
                len(self.category_labels['%scateid' % letter]))
        print('')
    
    def describe_features(self):
        print("columns of features:\n")
        for column in self.columns:
            print("-", column)
        print('')
    
    def remove_attrs(self, s):
        num = re.compile(r'^[-+]?([1-9]\d*|0)$')
        punc = re.compile('[%s]' % re.escape(string.punctuation))
        s = num.sub('', s)
        s = punc.sub('', s)
        return s

    def tokenize_feature_morphemes(self):
        morpheme_tokenizer = PosTokenizer(stem=True)

        for phase in self.phase:
            for column in self.text_feature_columns:
                print("tokenizing %s of %s phase" % (column, phase))                
                self.text_feature_maps[phase][column] = morpheme_tokenizer.fit_transform(
                    self.text_feature_maps[phase][column])
                
    def convert_feature_to_sequences(self, tokenizer_pickle="models/%s_tokenizer.pkl"):
        tokenizer = {
            'product': None,
            'model': None,
            'brand': None,
            'maker': None,
        }
        if not os.path.exists("./data/prep/x.pkl"):
            for category in self.text_feature_columns:
                if os.path.exists(tokenizer_pickle % category):
                    print("loading pre-generated tokenizer")
                    with open(tokenizer_pickle % category, 'rb') as handle:
                        tokenizer[category] = pickle.load(handle)
                else:
                    tokenizer[category] = Tokenizer(num_words=param.vocab_size)
                    tokenizer[category].fit_on_texts(pd.concat(
                        [self.text_feature_maps['train'][category],
                        self.text_feature_maps['dev'][category],
                        self.text_feature_maps['test'][category]]))

                    # saving tokenizer
                    with open(tokenizer_pickle % category, 'wb+') as handle:
                        pickle.dump(tokenizer[category], handle, protocol=pickle.HIGHEST_PROTOCOL)

            for phase in self.phase:
                for i, column in enumerate(self.text_feature_columns):
                    sequences = tokenizer[category].texts_to_sequences(
                        self.text_feature_maps[phase][column])
                    if i == 0:
                        self.x[phase] = pad_sequences(sequences, maxlen=param.max_length)
                    else:
                        self.x[phase] = np.concatenate(
                            [self.x[phase], pad_sequences(sequences, maxlen=param.max_length)],
                            axis=-1)
                        
            with open("./data/prep/x.pkl", "wb+") as handle:
                pickle.dump(self.x, handle, protocol=pickle.HIGHEST_PROTOCOL)        
        else:
            with open("./data/prep/x.pkl", "rb") as handle:
                self.x = pickle.load(handle)
                
        print("shape of the feature sequences:")
        print("train:", self.x['train'].shape)
        print("dev:", self.x['dev'].shape)
        print("test:", self.x['test'].shape)

    def initialize_label_encoders(self, exclude_unknown=True):
        self.encoders = {
            'b': LabelEncoder(),
            'm': LabelEncoder(),
            's': LabelEncoder(),
            'd': LabelEncoder()
        }

        # Fit label encoders to unique category labels
        self.encoders['b'].fit(sorted([*self.category_labels['bcateid'].keys()]))
        self.encoders['m'].fit(sorted([*self.category_labels['mcateid'].keys()]))

        if exclude_unknown:
            # Exclude unknown labels
            self.encoders['s'].fit(sorted([*self.category_labels['scateid'].keys()])[1:])
            self.encoders['d'].fit(sorted([*self.category_labels['dcateid'].keys()])[1:])
        else:
            self.train_y = {
                'b': self.train_dataset['bcateid'],
                'm': self.train_dataset['mcateid'],
                's': self.train_dataset['scateid'],
                'd': self.train_dataset['dcateid'],
            }
            self.y['train'] = self.train_y
            self.encoders['s'].fit(sorted([*self.category_labels['scateid'].keys()]))
            self.encoders['d'].fit(sorted([*self.category_labels['dcateid'].keys()]))
    
    def build_cnn_model(self, model_name='YoonKimCNNv3', category='b', 
                        embedding_size=param.embedding_size, visualize=True, without_dense=False):
        model = NETWORKS[model_name](dropout_rate=param.dropout_rate)

        sequence_input = layers.Input(shape=(param.max_length*4,), dtype='int32')
        x_embed = layers.Embedding(
            param.vocab_size, embedding_size,
            input_length=param.max_length*4, name="embedding")(sequence_input)
        x_embed = model(x_embed)
        
        if without_dense:
            return Model(sequence_input, x_embed)

        result = layers.Dense(len(self.encoders[category].classes_), activation='softmax',
                              name='classifier_%s' % category)(x_embed)
        model = Model(sequence_input, result)
        model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])

        if visualize:
            # Visualize model
            plot_model(model, to_file="models/model.png", show_shapes=True)

        return model
    
    def build_allinone_model(self, visualize=False):
        model = YoonKimCNNv2(dropout_rate=param.dropout_rate)
        sequence_input = layers.Input(shape=(param.max_length*4), dtype='int32')
        x_embed = layers.Embedding(
            param.vocab_size, param.embedding_size,
            input_length=param.max_length*4, name="embedding")(sequence_input)
        x_embed = model(x_embed)
        results = []
        for category in  self.categories:
            results.append(layers.Dense(len(self.encoders[category].classes_), activation='softmax',
                           name='dense_%s' % category)(x_embed))
        model = Model(sequence_input, results)
        model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])

        if visualize:
            # Visualize model
            plot_model(model, to_file="models/model.png", show_shapes=True)
            
        return model
    
    def build_han(self, category='b'):
        han = NETWORKS['HAN'](param.max_length, 4, len(self.encoders[category].classes_),
                              dense_layer_subfix=category)
        han.compile(loss='sparse_categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
        return han
    
    def train_at_once(self, test_split_ratio=0, epoch=param.epoch):
        self.initialize_label_encoders(exclude_unknown=False)
        y_train_b = self.y['train']['b']
        y_train_m = self.y['train']['m']
        y_train_s = self.y['train']['s']
        y_train_d = self.y['train']['d']
        
        y_train_b = self.encoders['b'].transform(y_train_b)
        y_train_m = self.encoders['m'].transform(y_train_m)
        y_train_s = self.encoders['s'].transform(y_train_s)
        y_train_d = self.encoders['d'].transform(y_train_d)
        
        model = self.build_allinone_model()
        checkpointer = ModelCheckpoint(
                "models/allinone.{epoch:02d}.hdf5", period=5)
        if test_split_ratio:
            x_train, x_test, y_train_b, y_test_b = train_test_split(
                self.x['train'], y_train_b, random_state=39)
            x_train, x_test, y_train_m, y_test_m = train_test_split(
                self.x['train'], y_train_m, random_state=39)
            x_train, x_test, y_train_s, y_test_s = train_test_split(
                self.x['train'], y_train_s, random_state=39)
            x_train, x_test, y_train_d, y_test_d = train_test_split(
                self.x['train'], y_train_d, random_state=39)
            
            y_train_dict = {
                "dense_b": y_train_b,
                "dense_m": y_train_m,
                "dense_s": y_train_s,
                "dense_d": y_train_d,
            }
            y_test_dict = {
                "dense_b": y_test_b,
                "dense_m": y_test_m,
                "dense_s": y_test_s,
                "dense_d": y_test_d,
            }
            
            model.fit(x=x_train, y=y_train_dict, 
                    batch_size=param.batch_size, epochs=epoch, 
                    verbose=1, validation_data=(x_test, y_test_dict),
                    callbacks=[checkpointer])
        else:
            y_train_dict = {
                "dense_b": y_train_b,
                "dense_m": y_train_m,
                "dense_s": y_train_s,
                "dense_d": y_train_d,
            }
            model.fit(x=self.x['train'], y=y_train_dict, 
                    batch_size=param.batch_size, epochs=epoch, 
                    verbose=1, callbacks=[checkpointer])

    def train(self, test_split_ratio=0, epoch=param.epoch, transfer=True,
              transfer_mode='prev', model_type='cnn', pseudo_label=False, 
              cnn_model_name='YoonKimCNNv2', categories=['b','m','s','d'],
              checkpoint_prefix='category', embedding_size=param.embedding_size):
        for i, category in enumerate(categories):
            
            if model_type == 'cnn':
                if pseudo_label:
                    print("using pseudo-label")
                    x_train = np.concatenate(
                        (self.x['train'],
                         self.x['dev'],
                         self.x['test'])) 
                    y_train = np.concatenate([
                        self.y['train'][category],
                        self.y['dev'][category],
                        self.y['test'][category]])
                else:
                    x_train, y_train = self.x['train'], self.y['train'][category]
            if model_type == 'han':
                # HAN Model은 3차원의 입력을 가짐
                x_train, y_train = self.x['train'].reshape(
                    len(self.x['train']), 4, param.max_length), self.y['train'][category]
                
            x_train = x_train[y_train != 0]        
            y_train = y_train[y_train != 0]
            y_train = self.encoders[category].transform(y_train)
            
            if model_type == 'cnn':
                model = self.build_cnn_model(category=category, 
                                             model_name=cnn_model_name, 
                                             embedding_size=embedding_size)
                if (i != 0) & transfer:
                    if transfer_mode == 'prev':
                        print("transfer learning using pre-trained network")
                        model.load_weights("models/%s_%s.%02d.hdf5" % (
                            categories[i-1], checkpoint_prefix, epoch), by_name=True)

                if transfer & (transfer_mode == 'allinone'):
                    print("transfer learning using pre-trained network")
                    model.load_weights("models/allinone.15.hdf5", by_name=True)
                    
                # Save model weights
                checkpointer = ModelCheckpoint(
                    "models/%s_%s.{epoch:02d}.hdf5" % (category, checkpoint_prefix), period=5)
                
            elif model_type == 'han':
                model = self.build_han(category)
                if (category != 'b') & transfer:
                    if transfer_mode == 'prev':
                        print("transfer learning using pre-trained network")
                        model.load_weights("models/%s_category_han.%02d.hdf5" % (
                            categories[i-1], epoch), by_name=True)

                if transfer & (transfer_mode != 'prev'):
                    print("transfer learning using pre-trained network")
                    model.load_weights("models/allinone.15.hdf5", by_name=True)

                # Save model weights
                checkpointer = ModelCheckpoint(
                    "models/%s_%s_han.{epoch:02d}.hdf5" % (category, checkpoint_prefix), period=1)

            if test_split_ratio:
                x_train, x_test, y_train, y_test = train_test_split(
                    x_train, y_train, random_state=39, test_size=test_split_ratio)
                model.fit(x=x_train, y=y_train, 
                        batch_size=param.batch_size, epochs=epoch, 
                        verbose=1, validation_data=(x_test, y_test),
                        callbacks=[checkpointer])
            else:
                model.fit(x=x_train, y=y_train, 
                        batch_size=param.batch_size, epochs=epoch, 
                        verbose=1, callbacks=[checkpointer])
            
    def predict(self, target='dev', epoch=param.epoch, batch_chunk_size=100000,
            model_name='YoonKimCNNv3', ensemble_mode='baseline',
            categories=['b','m','s','d'], category_w=2.0):

        if not isinstance(epoch, list):
            epoch = [epoch]
            
        for category in categories:
            models = []
            # Baseline model recipe
            if ensemble_mode == 'baseline':
                models.append(self.build_cnn_model(category=category,  model_name=model_name))
                models[0].load_weights("models/%s_category.%02d.hdf5" % (category, epoch[0]))

            # HAN recipe
            if ensemble_mode == 'han':
                models.append(self.build_han(category))
                models[0].load_weights("models/%s_category_han.%02d.hdf5" % (category, epoch[0])) 

            # Triple blending recipe
            # predict(epoch=[10,15,20])
            if ensemble_mode == 'triple':
                for i, e in enumerate(epoch):
                    if i < 2:
                        models.append(self.build_cnn_model(
                            category=category, model_name='YoonKimCNNv3', embedding_size=150))
                    else:
                        models.append(self.build_cnn_model(
                            category=category, model_name='YoonKimCNNv2', embedding_size=150))
                    models[i].load_weights("models/%s_triple.%02d.hdf5" % (category, e))
                        
            if ensemble_mode == 'quad':
                for i, e in enumerate(epoch):
                    if i < 2:
                        models.append(self.build_cnn_model(
                            category=category, model_name='YoonKimCNNv3', embedding_size=150))
                    elif i == 2:
                        models.append(self.build_cnn_model(
                            category=category, model_name='YoonKimCNNv2', embedding_size=150))
                    elif i >= 3:
                        models.append(self.build_cnn_model(
                            category=category, model_name='YoonKimCNN', embedding_size=200))
                    if i < 3:
                        models[i].load_weights("models/%s_triple.%02d.hdf5" % (category, e))
                    else:
                        models[i].load_weights("models/%s_category.%02d.hdf5" % (category, e))
                        
            if category == 'm':
                bcate_result = self.preds['b']
            if category == 's':
                bcate_result = self.preds['b']
                mcate_result = self.preds['m']
            if category == 'd':
                bcate_result = self.preds['b']
                mcate_result = self.preds['m']
                scate_result = self.preds['s']
                        
            if category == 'b':
                def predict_bcategory():
                    y_pred_ = np.array([])
                    print("predicting bcategory")
                    for batch in gen_batch(self.x[target], batch_chunk_size):
                        raw_predictions = []
                        for i in range(len(models)):
                            if 'han' in models[i].name:
                                batch = batch.reshape(len(batch), 4, param.max_length)
                            raw_predictions.append(models[i].predict(batch))
                        if ensemble_mode != 'quad':
                            raw_prediction = np.mean(raw_predictions, axis=0)
                        else:
                            raw_prediction_1 = np.mean(raw_predictions[:2], axis=0)
                            raw_prediction_2 = np.mean(raw_predictions[2:], axis=0)
                            raw_prediction = np.mean([raw_prediction_1, raw_prediction_2], axis=0)
                        y_pred_ = np.concatenate(
                            (y_pred_, np.argmax(raw_prediction, axis=1)), axis=-1)
                    return y_pred_
                y_pred_ = predict_bcategory()

            if category == 'm':
                def predict_mcategory():
                    y_pred_ = np.array([])
                    print("predicting mcategory")
                    # bcategory의 prediction 결과에 해당하는 raw_prediction
                    # index에 * 1.1의 weight를 곱함
                    for batch, b_batch in zip(gen_batch(self.x[target], batch_chunk_size),
                                              gen_batch(bcate_result, batch_chunk_size)):
                        b_weight_index = []
                        for b in b_batch:
                            b_weight_index.append(
                                self.encoders['m'].transform(parent_category['bcateid'][b].m))

                        raw_predictions = []
                        for i in range(len(models)):
                            if 'han' in models[i].name:
                                batch = batch.reshape(len(batch), 4, param.max_length)
                            raw_predictions.append(models[i].predict(batch))
                        if ensemble_mode != 'quad':
                            raw_prediction = np.mean(raw_predictions, axis=0)
                        else:
                            raw_prediction_1 = np.mean(raw_predictions[:2], axis=0)
                            raw_prediction_2 = np.mean(raw_predictions[2:], axis=0)
                            raw_prediction = np.mean([raw_prediction_1, raw_prediction_2], axis=0)
                        for i, b_w in zip(range(len(raw_prediction)), b_weight_index):
                            if b_w.size:
                                raw_prediction[i][b_w] *= category_w

                        y_pred_ = np.concatenate(
                            (y_pred_, np.argmax(raw_prediction, axis=1)), axis=-1)
                    return y_pred_
                y_pred_ = predict_mcategory()

            if category == 's':
                def predict_scategory():
                    y_pred_ = np.array([])
                    print("predicting scategory")
                    # mcategory, bcategory의 prediction 결과에 해당하는 raw_prediction
                    # index에 * 1.1의 weight를 곱함
                    for batch, b_batch, m_batch in zip(gen_batch(self.x[target], batch_chunk_size),
                                                       gen_batch(bcate_result, batch_chunk_size),
                                                       gen_batch(mcate_result, batch_chunk_size)):                    
                        b_weight_index = []
                        for b in b_batch:
                            b_weight_index.append(
                                self.encoders['s'].transform(parent_category['bcateid'][b].s))

                        m_weight_index = []
                        for m in m_batch:
                            m_weight_index.append(
                                self.encoders['s'].transform(parent_category['mcateid'][m].s))

                        raw_predictions = []
                        for i in range(len(models)):
                            if 'han' in models[i].name:
                                batch = batch.reshape(len(batch), 4, param.max_length)
                            raw_predictions.append(models[i].predict(batch))
                        if ensemble_mode != 'quad':
                            raw_prediction = np.mean(raw_predictions, axis=0)
                        else:
                            raw_prediction_1 = np.mean(raw_predictions[:2], axis=0)
                            raw_prediction_2 = np.mean(raw_predictions[2:], axis=0)
                            raw_prediction = np.mean([raw_prediction_1, raw_prediction_2], axis=0)
                        for i, b_w, m_w in zip(range(len(raw_prediction)),
                                               b_weight_index, m_weight_index):
                            if b_w.size:
                                raw_prediction[i][b_w] *= category_w
                            if m_w.size:
                                raw_prediction[i][m_w] *= category_w

                        y_pred_ = np.concatenate(
                            (y_pred_, np.argmax(raw_prediction, axis=1)), axis=-1)
                    return y_pred_
                y_pred_ = predict_scategory()
                    
            if category == 'd':
                def predict_dcategory():
                    y_pred_ = np.array([])
                    print("predicting dcategory")
                    # scategory, mcategory, bcategory의 prediction 결과에 해당하는 raw_prediction
                    # index에 * 1.1의 weight를 곱함
                    for batch, b_batch, m_batch, s_batch in zip(gen_batch(self.x[target], batch_chunk_size),
                                                                gen_batch(bcate_result, batch_chunk_size),
                                                                gen_batch(mcate_result, batch_chunk_size),
                                                                gen_batch(scate_result, batch_chunk_size)):                    
                        b_weight_index = []
                        for b in b_batch:
                            b_weight_index.append(
                                self.encoders['d'].transform(parent_category['bcateid'][b].d))

                        m_weight_index = []
                        for m in m_batch:
                            m_weight_index.append(
                                self.encoders['d'].transform(parent_category['mcateid'][m].d))

                        s_weight_index = []
                        for s in s_batch:
                            s_weight_index.append(
                                self.encoders['d'].transform(parent_category['scateid'][s].d))

                        raw_predictions = []
                        for i in range(len(models)):
                            if 'han' in models[i].name:
                                batch = batch.reshape(len(batch), 4, param.max_length)
                            raw_predictions.append(models[i].predict(batch))
                        if ensemble_mode != 'quad':
                            raw_prediction = np.mean(raw_predictions, axis=0)
                        else:
                            raw_prediction_1 = np.mean(raw_predictions[:2], axis=0)
                            raw_prediction_2 = np.mean(raw_predictions[2:], axis=0)
                            raw_prediction = np.mean([raw_prediction_1, raw_prediction_2], axis=0)
                        for i, b_w, m_w, s_w in zip(range(len(raw_prediction)),
                                                    b_weight_index, m_weight_index, s_weight_index):
                            if b_w.size:
                                raw_prediction[i][b_w] *= category_w
                            if m_w.size:
                                raw_prediction[i][m_w] *= category_w
                            if s_w.size:
                                raw_prediction[i][s_w] *= category_w

                        y_pred_ = np.concatenate(
                            (y_pred_, np.argmax(raw_prediction, axis=1)), axis=-1)
                    return y_pred_
                y_pred_ = predict_dcategory()
                    
            y_pred = self.encoders[category].inverse_transform(
                np.array(y_pred_).astype(int))  # Decode prediction

            self.preds[category] = y_pred

        if target == 'train':
            self.pred_to_tsv(
                self.train_dataset['pid'], self.preds,
                filename="train_result.tsv")
        if target == 'dev':
            self.pred_to_tsv(
                self.dev_dataset['pid'], self.preds,
                filename="dev_result.tsv")
        if target == 'test':
            self.pred_to_tsv(
                self.test_dataset['pid'], self.preds,
                filename="test_result.tsv")

    @staticmethod
    def pred_to_tsv(pid, pred, filename="result.tsv"):
        bcate_pred = pred['b']
        mcate_pred = pred['m']
        scate_pred = pred['s']
        dcate_pred = pred['d']
        template = "{pid}\t{b}\t{m}\t{s}\t{d}"
        
        with open(filename, 'w+') as tsv: 
            for current_pid, b, m, s, d in zip(
                pid, bcate_pred, mcate_pred, scate_pred, dcate_pred):
                if s == 0:
                    s = -1
                if d == 0:
                    d = -1
                line = template.format(
                    pid=current_pid, b=int(b), m=int(m), s=int(s), d=int(d))
                tsv.write(line)
                tsv.write('\n')
                
    def run(self, phase='train_and_predict', test_split_ratio=0,
            predict_target='dev'):
        self.describe_features()
        self.describe_labels()
        if os.path.exists("./data/prep/x.pkl"):
            with open("./data/prep/x.pkl", "rb") as handle:
                self.x = pickle.load(handle)
        else:
            self.initialize_features()
            self.tokenize_feature_morphemes()
            self.convert_feature_to_sequences()
        self.initialize_label_encoders()

        if phase == 'train':
            self.train(test_split_ratio)

        if phase == 'predict':
            self.predict(target=predict_target)
        
        if phase == 'blending_train':
            self.train(transfer_mode='prev', epoch=20, model_type='cnn',
                       cnn_model_name='YoonKimCNNv2', checkpoint_prefix='triple')
            self.train_at_once(epoch=15)
            self.train(transfer_mode='allinone', epoch=15, model_type='cnn', 
                       cnn_model_name='YoonKimCNNv3', checkpoint_prefix='triple')
            self.train(transfer_mode='prev', epoch=20, model_type='cnn',
                       cnn_model_name='YoonKimCNN', checkpoint_prefix='category')

        if phase == 'blending_predict':
            self.predict(epoch=[10,15,20,15,20], ensemble_mode='quad',
                         category_w=2.0, target=predict_target)
            

if __name__ == '__main__':
    import fire
    classifier = TextFeatureClassifier()
    fire.Fire(classifier.run)