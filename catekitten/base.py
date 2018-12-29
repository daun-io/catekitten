from tensorflow.keras import layers


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


class ConsumeMask(layers.Layer):
    """Layer that prevents mask propagation.
    """

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        return x

