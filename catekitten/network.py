from tensorflow.keras import layers
from catekitten.base import SequenceEncoderBase
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from scipy import sparse

import tensorflow as tf
import numpy as np


class NbSvmClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, dual=False, n_jobs=1):
        self.C = C
        self.dual = dual
        self.n_jobs = n_jobs

    def predict(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict(x.multiply(self._r))

    def predict_proba(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict_proba(x.multiply(self._r))

    def fit(self, x, y):
        # Check that X and y have correct shape
        x, y = check_X_y(x, y, accept_sparse=True)
        
        if isinstance(x, np.ndarray):
            x = sparse.csr_matrix(x)
            
        def pr(x, y_i, y):
            p = x[y==y_i].sum(0)
            return (p+1) / ((y==y_i).sum()+1)

        self._r = sparse.csr_matrix(np.log(pr(x,1,y) / pr(x,0,y)))
        x_nb = x.multiply(self._r)
        self._clf = LogisticRegression(C=self.C, dual=self.dual, n_jobs=self.n_jobs).fit(x_nb, y)
        return self


class BidirectionalCNN(SequenceEncoderBase):
    def __init__(self, num_filters=64, dropout_rate=0.5):
        """Modified version of Yoon Kim's shallow cnn model: https://arxiv.org/pdf/1408.5882.pdf
        Args:
            num_filters: The number of filters to use per `filter_size`. (Default value = 64)
            filter_sizes: The filter sizes for each convolutional layer. (Default value = [3, 4, 5])
            **cnn_kwargs: Additional args for building the `Conv1D` layer.
        """
        super(BidirectionalCNN, self).__init__(dropout_rate)
        self.num_filters = num_filters

    def build_model(self, x):
        x = layers.Bidirectional(layers.GRU(self.num_filters, return_sequences=True))(x)
        x = layers.Conv1D(64, kernel_size=3, padding="valid", kernel_initializer="glorot_uniform")(x)
        x = layers.GlobalMaxPooling1D()(x)
        return x


class YoonKimCNNv3(SequenceEncoderBase):
    def __init__(self, num_filters=128, filter_sizes=[2, 3, 4, 5], dropout_rate=0.5, **conv_kwargs):
        """Modified version of Yoon Kim's shallow cnn model: https://arxiv.org/pdf/1408.5882.pdf
        Args:
            num_filters: The number of filters to use per `filter_size`. (Default value = 64)
            filter_sizes: The filter sizes for each convolutional layer. (Default value = [3, 4, 5])
            **cnn_kwargs: Additional args for building the `Conv1D` layer.
        """
        super(YoonKimCNNv3, self).__init__(dropout_rate)
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.conv_kwargs = conv_kwargs

    def build_model(self, x):
        maxpooled_tensors = []
        avgpooled_tensors = []
        for filter_size in self.filter_sizes:
            x_i = layers.Conv1D(self.num_filters, filter_size,
                                use_bias=False, **self.conv_kwargs,
                                name="conv%s" % filter_size)(x)
            x_i = layers.ELU(name="elu%s" % filter_size)(x_i)
            x_i = layers.BatchNormalization(name="bn%s" % filter_size)(x_i)
            x_m = layers.GlobalMaxPooling1D(name="global_maxpool%s" % filter_size)(x_i)
            x_a = layers.GlobalAveragePooling1D(name="global_avgpool%s" % filter_size)(x_i)
            maxpooled_tensors.append(x_m)
            avgpooled_tensors.append(x_a)
        x_m = layers.concatenate(maxpooled_tensors, axis=-1)
        x_a = layers.concatenate(avgpooled_tensors, axis=-1)
        x = layers.concatenate([x_m, x_a], axis=-1)
        return x


class YoonKimCNNv2(SequenceEncoderBase):
    def __init__(self, num_filters=128, filter_sizes=[2, 3, 4, 5], dropout_rate=0.5, **conv_kwargs):
        """Modified version of Yoon Kim's shallow cnn model: https://arxiv.org/pdf/1408.5882.pdf
        Args:
            num_filters: The number of filters to use per `filter_size`. (Default value = 64)
            filter_sizes: The filter sizes for each convolutional layer. (Default value = [3, 4, 5])
            **cnn_kwargs: Additional args for building the `Conv1D` layer.
        """
        super(YoonKimCNNv2, self).__init__(dropout_rate)
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.conv_kwargs = conv_kwargs

    def build_model(self, x):
        pooled_tensors = []
        for filter_size in self.filter_sizes:
            x_i = layers.Conv1D(self.num_filters, filter_size,
                                use_bias=False, **self.conv_kwargs,
                                name="conv%s" % filter_size)(x)
            x_i = layers.ELU(name="elu%s" % filter_size)(x_i)
            x_i = layers.BatchNormalization(name="bn%s" % filter_size)(x_i)
            x_i = layers.GlobalMaxPooling1D(name="global_maxpool%s" % filter_size)(x_i)
            pooled_tensors.append(x_i)
        x = pooled_tensors[0] if len(self.filter_sizes) == 1 else layers.concatenate(pooled_tensors, axis=-1)
        return x


class YoonKimCNN(SequenceEncoderBase):
    def __init__(self, num_filters=128, filter_sizes=[3, 4, 5], dropout_rate=0.5, **conv_kwargs):
        """Modified version of Yoon Kim's shallow cnn model: https://arxiv.org/pdf/1408.5882.pdf
        Args:
            num_filters: The number of filters to use per `filter_size`. (Default value = 64)
            filter_sizes: The filter sizes for each convolutional layer. (Default value = [3, 4, 5])
            **cnn_kwargs: Additional args for building the `Conv1D` layer.
        """
        super(YoonKimCNN, self).__init__(dropout_rate)
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.conv_kwargs = conv_kwargs

    def build_model(self, x):
        pooled_tensors = []
        for filter_size in self.filter_sizes:
            x_i = layers.Conv1D(self.num_filters, filter_size, use_bias=False, **self.conv_kwargs)(x)
            x_i = layers.ELU()(x_i)
            x_i = layers.BatchNormalization()(x_i)
            x_i = layers.GlobalMaxPooling1D()(x_i)
            pooled_tensors.append(x_i)
        x = pooled_tensors[0] if len(self.filter_sizes) == 1 else layers.concatenate(pooled_tensors, axis=-1)
        return x


class TextLab(SequenceEncoderBase):
    def __init__(self, num_filters=128, filter_sizes=[3, 4, 5], dropout_rate=0.5, **conv_kwargs):
        """Modified version of Yoon Kim's shallow cnn model: https://arxiv.org/pdf/1408.5882.pdf
        Args:
            num_filters: The number of filters to use per `filter_size`. (Default value = 64)
            filter_sizes: The filter sizes for each convolutional layer. (Default value = [3, 4, 5])
            **cnn_kwargs: Additional args for building the `Conv1D` layer.
        """
        super(TextLab, self).__init__(dropout_rate)
        self.filter_sizes = filter_sizes
        self.conv_kwargs = conv_kwargs
        self.num_filters = num_filters

    def conv_block(self, x, num_filters, num_kernels=2):
        bypass = x

        for _ in range(num_kernels):
            x = layers.Conv1D(num_filters, 3, padding='same',
                              activation='relu', use_bias=False, **self.conv_kwargs)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.dropout_rate)(x)

        x = layers.Add()([bypass, x])
        return x

    def build_model(self, x):
        pooled_tensors = []

        with tf.variable_scope("encoder"):
            for filter_size in self.filter_sizes:
                x_i = layers.Conv1D(self.num_filters, filter_size, use_bias=False, **self.conv_kwargs)(x)
                x_i = layers.BatchNormalization()(x_i)
                x_i = layers.GlobalMaxPooling1D()(x_i)
                pooled_tensors.append(x_i)
            x = pooled_tensors[0] if len(self.filter_sizes) == 1 else layers.concatenate(pooled_tensors, axis=-1)
        
        with tf.variable_scope("decoder"):
            x = self.conv_block(x, self.num_filters)
            
        return x
