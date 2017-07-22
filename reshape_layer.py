from keras import backend as K
import tensorflow as tf
from keras.engine.topology import Layer
import numpy as np

class custom_reshape(Layer):

    def __init__(self, target_shape, **kwargs):
        super(custom_reshape, self).__init__(**kwargs)
        self.target_shape = tuple(target_shape)

    def _fix_unknown_dimension(self, input_shape, output_shape):
        """Finds and replaces a missing dimension in an output shape.
        This is a near direct port of the internal Numpy function
        `_fix_unknown_dimension` in `numpy/core/src/multiarray/shape.c`
        # Arguments
            input_shape: original shape of array being reshaped
            output_shape: target shape of the array, with at most
                a single -1 which indicates a dimension that should be
                derived from the input shape.
        # Returns
            The new output shape with a `-1` replaced with its computed value.
        # Raises
            ValueError: if `input_shape` and `output_shape` do not match.
        """
        output_shape = list(output_shape)
        msg = 'total size of new array must be unchanged'

        known, unknown = 1, None
        for index, dim in enumerate(output_shape):
            if dim < 0:
                if unknown is None:
                    unknown = index
                else:
                    raise ValueError('Can only specify one unknown dimension.')
            else:
                known *= dim

        original = np.prod(input_shape, dtype=int)
        if unknown is not None:
            if known == 0 or original % known != 0:
                raise ValueError(msg)
            output_shape[unknown] = original // known
        elif original != known:
            raise ValueError(msg)

        return tuple(output_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + self._fix_unknown_dimension(
            input_shape[1:], self.target_shape)

    def call(self, inputs):
        # In case the target shape is not fully defined,
        # we need access to the shape of `inputs`.
        # solution: rely on `K.int_shape`.
        target_shape = self.target_shape
        if -1 in target_shape:
            # Target shape not fully defined.
            input_shape = None
            try:
                input_shape = K.int_shape(inputs)
            except TypeError:
                pass
            if input_shape is not None:
                target_shape = self.compute_output_shape(input_shape)[1:]
        split = tf.split(inputs, input_shape[3], axis=3)
        c = tf.concat(split, axis=1)
        reshaped = tf.squeeze(c, axis=-1)
        reshaped = tf.transpose(reshaped, [0,2,1])
        return reshaped

    def get_config(self):
        config = {'target_shape': self.target_shape}
        base_config = super(custom_reshape, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))