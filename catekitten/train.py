from catekitten.data import Loader
from catekitten.data import get_category_map
from catekitten.transform import PosTokenizer
from catekitten.utils import lowercase, gen_batch
from catekitten.network import YoonKimCNN
from catekitten import param

import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import os

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


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

    def describe_labels(self):
        for letter in ['b','m','s','d']:
            print("number of %scateid classes:", 
                len(self.category_labels['%scateid' % letter]))
    
    def describe_features(self):
        print("columns of features:")
        for column in self.columns:
            print(column)

    def tokenize_feature_morphemes(self):
        morpheme_tokenizer = PosTokenizer(stem=True)

        for phase in self.phase:
            for column in self.text_feature_columns:
                print("tokenizing %s of %s phase" % (column, phase))
                self.text_feature_maps[phase][column].map(lowercase)
                self.text_feature_maps[phase][column] = morpheme_tokenizer.fit_transform(
                    self.text_feature_maps[column])

    def convert_feature_to_sequences(self, tokenizer_pickle="models/tokenizer.pkl"):
        if os.path.exists(tokenizer_pickle):
            print("loading pre-generated tokenizer")
            with open(tokenizer_pickle, 'rb') as handle:
                tokenizer = pickle.load(handle)
        else:
            tokenizer = Tokenizer(num_words=param.vocab_size)
            tokenizer.fit_on_texts(pd.concat(
                [self.text_feature_maps['train'],
                self.text_feature_maps['dev'],
                self.text_feature_maps['test']]))

            # saving tokenizer
            with open(tokenizer_pickle, 'wb') as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        for phase in self.phase:
            for i, column in enumerate(self.text_feature_columns):
                sequences = tokenizer.text_to_sequences(
                    self.text_feature_maps[phase][column])
                if i == 0:
                    self.x[phase] = pad_sequences(sequences, maxlen=param.max_length)
                else:
                    self.x[phase] = np.concatenate(
                        [self.x[phase], pad_sequences(sequences, maxlen=param.max_length)],
                        axis=-1)

        print("shape of the feature sequences:")
        print("train:", self.x['train'].shape)
        print("dev:", self.x['dev'].shape)
        print("test:", self.x['test'].shape)

    def initialize_label_encoders(self):
        self.encoders = {
            'b': LabelEncoder(),
            'm': LabelEncoder(),
            's': LabelEncoder(),
            'd': LabelEncoder()
        }

        # Fit label encoders to unique category labels
        self.encoders['b'].fit(sorted([*self.category_labels['bcateid'].keys()]))
        self.encoders['m'].fit(sorted([*self.category_labels['mcateid'].keys()]))

        # Exclude unknown labels
        self.encoders['s'].fit(sorted([*self.category_labels['scateid'].keys()])[1:])
        self.encoders['d'].fit(sorted([*self.category_labels['dcateid'].keys()])[1:])

    def build_model(self, category='b'):
        model = YoonKimCNN(dropout_rate=param.dropout_rate)

        sequence_input = layers.Input(shape=(param.max_length*4,), dtype='int32')
        x_embed = layers.Embedding(
            param.vocab_size, param.embedding_size, input_length=param.max_length*4)(sequence_input)
        x_embed = model(x_embed)

        result = layers.Dense(len(self.encoders[category].classes_), activation='softmax')(x_embed)

        model = Model(sequence_input, result)
        model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])

        # Visualize model
        plot_model(model, to_file="models/model.png", show_shapes=True)

        return model

    def train(self, test_split_ratio=0):
        for category in self.categories:
            x_train, y_train = self.x['train'], self.y['train'][category]
            x_train = x_train[y_train != 0]        
            y_train = y_train[y_train != 0]
            y_train = self.encoders[category].transform(y_train)

            model = self.build_model(category)

            # Save model weights
            checkpointer = ModelCheckpoint(
                "models/%s_category.{epoch:02d}.hdf5" % category)

            if test_split_ratio:
                x_train, x_test, y_train, y_test = train_test_split(
                    x_train, y_train, random_state=39)
                model.fit(x=x_train, y=y_train, 
                        batch_size=param.batch_size, epochs=param.epoch, 
                        verbose=1, validation_data=(x_test, y_test),
                        callbacks=[checkpointer])
            else:
                model.fit(x=x_train, y=y_train, 
                        batch_size=param.batch_size, epochs=param.epoch, 
                        verbose=1, callbacks=[checkpointer])

    def predict(self, target='dev', epoch=param.epoch):
        for category in self.categories:
            model = self.build_model(category)
            model.load_weights("models/%s_category.%02d.hdf5" % (category, epoch))

            # Batch prediction
            y_pred = np.array([])
            for batch in gen_batch(self.x['dev'], 100000):
                y_pred = np.concatenate(
                    (y_pred, np.argmax(model.predict(batch), axis=1)), axis=-1)

                y_pred = self.encoders[category].inverse_transform(
                    np.array(y_pred).astype(int))  # Decode prediction

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
        self.tokenize_feature_morphemes()
        self.convert_feature_to_sequences()
        self.initialize_label_encoders()

        if 'train' in phase:
            self.train(test_split_ratio)

        if 'predict' in phase:
            self.predict(target=predict_target)


if __name__ == '__main__':
    import fire
    classifier = TextFeatureClassifier()
    fire.Fire(classifier.run)