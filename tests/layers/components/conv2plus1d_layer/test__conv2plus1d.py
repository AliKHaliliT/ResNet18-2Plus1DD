import unittest
from ResNet182Plus1DD.assets.layers.components.conv2plus1d_layer._conv2plus1d import _Conv2Plus1D
import tensorflow as tf
import inspect


class Test_Conv2Plus1D(unittest.TestCase):

    def test_filters_wrong__value_value__error(self):
        
        # Arrange
        filters = -1

        # Act and Assert
        with self.assertRaises(ValueError):
            _Conv2Plus1D(filters=filters, kernel_size=(1, 1, 1), strides=(1, 1, 1), 
                         padding="same")


    def test_kernel__size_wrong__value_value__error(self):

        # Arrange
        kernel_size = (-1, -1, -1)

        # Act and Assert
        with self.assertRaises(ValueError):
            _Conv2Plus1D(filters=1, kernel_size=kernel_size, strides=(1, 1, 1), 
                         padding="same")

    
    def test_strides_wrong__value_value__error(self):
        
        # Arrange
        strides = (-1, -1, -1)

        # Act and Assert
        with self.assertRaises(ValueError):
            _Conv2Plus1D(filters=1, kernel_size=(1, 1, 1), strides=strides, 
                         padding="same")

    
    def test_padding_wrong__value_value__error(self):
        
        # Arrange
        padding = "test"

        # Act and Assert
        with self.assertRaises(ValueError):
            _Conv2Plus1D(filters=1, kernel_size=(1, 1, 1), strides=(1, 1, 1), 
                         padding=padding)


    def test_use__bias_wrong__type_type__error(self):
        
        # Arrange
        use_bias = None

        # Act and Assert
        with self.assertRaises(TypeError):
            _Conv2Plus1D(filters=1, kernel_size=(1, 1, 1), strides=(1, 1, 1), 
                         padding="same", use_bias=use_bias)


    def test_creation_init_layer(self):

        # Arrange and Act
        layer = _Conv2Plus1D(filters=1, kernel_size=(1, 1, 1), strides=(1, 1, 1), 
                             padding="same")

        # Assert
        self.assertTrue(isinstance(layer, tf.keras.layers.Layer))


    def test_build_input__shape_layer(self):

        # Arrange
        layer = _Conv2Plus1D(filters=1, kernel_size=(1, 1, 1), strides=(1, 1, 1), 
                             padding="same")
        input_shape = (None, 1, 1, 1, 3)

        # Act
        layer.build(input_shape=input_shape)

        # Assert
        self.assertTrue(layer.built)


    def test_output_tensor_intended__shape(self):

        # Arrange
        input_tensor = tf.random.normal((1, 1, 1, 1, 3))

        # Act
        output = _Conv2Plus1D(filters=1, kernel_size=(1, 1, 1), strides=(1, 1, 1), 
                              padding="same")(input_tensor)

        # Assert
        self.assertEqual(output.shape, (1, 1, 1, 1, 1))


    def test_compute__output__shape_shape_intended__shape(self):

        # Arrange
        layer = _Conv2Plus1D(filters=1, kernel_size=(1, 1, 1), strides=(1, 1, 1), 
                             padding="same")
        input_tensor = tf.random.normal((1, 1, 1, 1, 3))
        
        # Act
        output = layer(input_tensor)
        output_shape = layer.compute_output_shape(input_tensor.shape)

        # Assert
        self.assertEqual(output.shape, output_shape)


    def test_get__config_init_matching__dict(self):

        # Arrange
        layer = _Conv2Plus1D(filters=1, kernel_size=(1, 1, 1), strides=(1, 1, 1), 
                             padding="same")

        # Act
        init_params = [
            param.name
            for param in inspect.signature(_Conv2Plus1D.__init__).parameters.values()
            if param.name != "self" and param.name != "kwargs" 
        ]

        # Assert
        self.assertTrue(all(param in layer.get_config() for param in init_params), "Missing parameters in get_config.")


if __name__ == "__main__":
    unittest.main()