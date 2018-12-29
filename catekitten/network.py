from tensorflow.keras import layers
import tensorflow as tf


# Keras model classes
class SequenceEncoderBase(object):
    
    def __init__(self, dropout_rate=0.5):
        """Creates a new instance of sequence encoder.
        Args:
            dropout_rate: The final encoded output dropout.
        """
        self.dropout_rate = dropout_rate

    def __call__(self, x):
        """Build the actual model here.
        Args:
            x: The encoded or embedded input sequence.
        Returns:
            The model output tensor.
        """
        # Avoid mask propagation when dynamic mini-batches are not supported.
        if not self.allows_dynamic_length():
            x = ConsumeMask()(x)

        x = self.build_model(x)
        if self.dropout_rate > 0:
            x = layers.Dropout(self.dropout_rate)(x)
        return x

    def build_model(self, x):
        """Build your model graph here.
        Args:
            x: The encoded or embedded input sequence.
        Returns:
            The model output tensor without the classification block.
        """
        raise NotImplementedError()

    def allows_dynamic_length(self):
        """Return a boolean indicating whether this model is capable of handling variable time steps per mini-batch.
        For example, this should be True for RNN models since you can use them with variable time steps per mini-batch.
        CNNs on the other hand expect fixed time steps across all mini-batches.
        """
        # Assume default as False. Should be overridden as necessary.
        return False


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


class ConsumeMask(layers.Layer):
    """Layer that prevents mask propagation.
    """

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        return x
    