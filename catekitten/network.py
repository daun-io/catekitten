from tensorflow.keras import layers
from catekitten.base import SequenceEncoderBase
import tensorflow as tf


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
