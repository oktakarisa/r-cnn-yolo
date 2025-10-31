try:
    from keras.engine import Layer, InputSpec
except ImportError:
    from keras.layers import Layer
    from keras import backend as K
    try:
        from keras.engine.keras_tensor import KerasTensor
        # New Keras version - InputSpec is different
        class InputSpec:
            def __init__(self, shape=None, ndim=None, dtype=None):
                self.shape = shape
                self.ndim = ndim
                self.dtype = dtype
    except ImportError:
        InputSpec = K.__dict__.get('InputSpec', type('InputSpec', (), {}))

from keras import initializers, regularizers
from keras import backend as K


class FixedBatchNormalization(Layer):

    def __init__(self, epsilon=1e-3, axis=-1,
                 weights=None, beta_init='zero', gamma_init='one',
                 gamma_regularizer=None, beta_regularizer=None, **kwargs):

        self.supports_masking = True
        self.beta_init = initializers.get(beta_init)
        self.gamma_init = initializers.get(gamma_init)
        self.epsilon = epsilon
        self.axis = axis
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.initial_weights = weights
        super(FixedBatchNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        # Handle InputSpec compatibility for different Keras versions
        try:
            # Try new Keras format
            from tensorflow.keras.layers import InputSpec
            self.input_spec = InputSpec(shape=input_shape)
        except (ImportError, TypeError):
            try:
                # Try old Keras format with list
                self.input_spec = [InputSpec(shape=input_shape)]
            except TypeError:
                # If InputSpec doesn't accept arguments, just set shape directly
                self.input_spec = InputSpec()
                self.input_spec.shape = input_shape
        
        shape = (input_shape[self.axis],)

        self.gamma = self.add_weight(shape,
                                     initializer=self.gamma_init,
                                     regularizer=self.gamma_regularizer,
                                     name='{}_gamma'.format(self.name),
                                     trainable=False)
        self.beta = self.add_weight(shape,
                                    initializer=self.beta_init,
                                    regularizer=self.beta_regularizer,
                                    name='{}_beta'.format(self.name),
                                    trainable=False)
        self.running_mean = self.add_weight(shape, initializer='zero',
                                            name='{}_running_mean'.format(self.name),
                                            trainable=False)
        self.running_std = self.add_weight(shape, initializer='one',
                                           name='{}_running_std'.format(self.name),
                                           trainable=False)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

        self.built = True

    def call(self, x, mask=None):

        assert self.built, 'Layer must be built before being called'
        # Handle both old and new Keras versions
        try:
            input_shape = K.int_shape(x)
        except AttributeError:
            # New Keras uses x.shape directly
            input_shape = x.shape.as_list() if hasattr(x.shape, 'as_list') else list(x.shape)

        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        if sorted(reduction_axes) == range(K.ndim(x))[:-1]:
            x_normed = K.batch_normalization(
                x, self.running_mean, self.running_std,
                self.beta, self.gamma,
                epsilon=self.epsilon)
        else:
            # need broadcasting
            broadcast_running_mean = K.reshape(self.running_mean, broadcast_shape)
            broadcast_running_std = K.reshape(self.running_std, broadcast_shape)
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            x_normed = K.batch_normalization(
                x, broadcast_running_mean, broadcast_running_std,
                broadcast_beta, broadcast_gamma,
                epsilon=self.epsilon,axis=0)

        return x_normed

    def compute_output_shape(self, input_shape):
        """For compatibility with older Keras"""
        return input_shape

    def compute_output_spec(self, inputs, **kwargs):
        """For newer Keras versions - mask is passed in kwargs but not used"""
        return inputs

    def get_config(self):
        config = {'epsilon': self.epsilon,
                  'axis': self.axis,
                  'gamma_regularizer': self.gamma_regularizer.get_config() if self.gamma_regularizer else None,
                  'beta_regularizer': self.beta_regularizer.get_config() if self.beta_regularizer else None}
        base_config = super(FixedBatchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
