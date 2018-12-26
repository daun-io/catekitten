import numpy as np
import h5py
import time
import datetime
import pickle
import gzip
import os

from catekitten.data import string_to_timestamp
from catekitten.evaluate import arena_accuracy_score
from catboost import CatBoostClassifier

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


class HierarchicalClassifier(object):
    def __init__(self, model=RandomForestClassifier):
        self.data = h5py.File('a:/git/shopping-classification/train.chunk.01')['train']

        # Initialize 4 different hierarchical y
        self.y_b = self.data['bcateid'][:400000]
        self.y_m = self.data['mcateid'][:400000]
        self.y_s = self.data['scateid'][:400000]
        self.y_d = self.data['dcateid'][:400000]

        # Initialize two simple modality
        self.x_price = self.data['price'][:400000]
        self.x_time = self.data['updttm'][:400000]

        # Convert datetime to unix timestamp to convert it into a continuous feature
        for i, x_t in enumerate(self.x_time):
            self.x_time[i] = int(string_to_timestamp(x_t.decode()))

        # Initialize feature
        self.x = np.zeros((400000, 2))
        self.x[:,0] = self.x_price
        self.x[:,1] = self.x_time

        # Our simple classifier model
        self.model = model
    
    def _save_model(self, model, model_filename):
        print("saving model to: %s" % model_filename)
        pickle.dump(model, gzip.open(model_filename, 'wb'))
    
    def _load_model(self, model_filename):
        print("loading model from: %s" % model_filename)
        return pickle.load(gzip.open(model_filename, 'rb'))
        
    def _train_vanilla_classifiers(self, model_dir="./model/"):
        #TODO: Partial Fit과 h5py를 활용해 batch training
        gts = [self.y_b, self.y_m, self.y_s, self.y_d]

        # Iteratively train and save multiple vanilla classifiers 
        for gt, letter in zip(gts, ['b','m','s','d']):
            x_train, x_test, y_train, y_test = train_test_split(self.x, gt, random_state=39)

            # Fit vanilla classifier
            classifier = self.model() 
            classifier.fit(x_train, y_train)

            # Save vanilla classifier
            model_filename = os.path.join(model_dir, "vanilla_%s.classifier" % letter)
            self._save_model(classifier, model_filename)

            # Predict and evaluate accuracy
            y_pred = classifier.predict(x_test)
            acc = accuracy_score(y_test, y_pred)
            print("vanilla %s model accuracy: %s" % (letter, acc))

    def _retrain_hierarchical_classifiers(self, model_dir="./model/"):
        gts = [self.y_b, self.y_m, self.y_s, self.y_d]

        # Iteratively train and save multiple hierarchical classifiers 
        for gt, i_letter in zip(gts, ('b','m','s','d')):
            x_train_orig, x_test_orig, y_train, y_test = train_test_split(self.x, gt, random_state=39)
            x_train, x_test = np.copy(x_train_orig), np.copy(x_test_orig)

            if i_letter == 'b':
                # for j_letter in ('m','s','d'):
                for j_letter in ('m'):
                    # Predict with minimal classifier
                    model_filename = os.path.join(model_dir, "vanilla_%s.classifier" % j_letter)
                    classifier = self._load_model(model_filename)
                    x_train = np.concatenate([x_train, classifier.predict_proba(x_train_orig)], axis=-1)
                    x_test = np.concatenate([x_test, classifier.predict_proba(x_test_orig)], axis=-1)

            if i_letter == 'm':
                # for j_letter in ('b','s','d'):
                for j_letter in ('b'):
                    # Predict with minimal classifier
                    model_filename = os.path.join(model_dir, "vanilla_%s.classifier" % j_letter)
                    classifier = self._load_model(model_filename)
                    x_train = np.concatenate([x_train, classifier.predict_proba(x_train_orig)], axis=-1)
                    x_test = np.concatenate([x_test, classifier.predict_proba(x_test_orig)], axis=-1)

            if i_letter == 's':
                # for j_letter in ('b','m','d'):
                for j_letter in ('m'):
                    # Predict with minimal classifier
                    model_filename = os.path.join(model_dir, "vanilla_%s.classifier" % j_letter)
                    classifier = self._load_model(model_filename)
                    x_train = np.concatenate([x_train, classifier.predict_proba(x_train_orig)], axis=-1)
                    x_test = np.concatenate([x_test, classifier.predict_proba(x_test_orig)], axis=-1)

            if i_letter == 'd':
                # for j_letter in ('b','m','s'):
                for j_letter in ('s'):
                    # Predict with minimal classifier
                    model_filename = os.path.join(model_dir, "vanilla_%s.classifier" % j_letter)
                    classifier = self._load_model(model_filename)
                    x_train = np.concatenate([x_train, classifier.predict_proba(x_train_orig)], axis=-1)
                    x_test = np.concatenate([x_test, classifier.predict_proba(x_test_orig)], axis=-1)

            classifier = self.model()
            classifier.fit(x_train, y_train)
            y_pred = classifier.predict(x_test)
            acc = accuracy_score(y_test, y_pred)
            print("hierarchical %s model accuracy: %s" % (i_letter, acc))

    def run_experiment(self):
        self._train_vanilla_classifiers()
        # self._retrain_hierarchical_classifiers()


if  __name__ == '__main__':
    model = HierarchicalClassifier()
    model.run_experiment()