import torch
from torch import nn

SHAPE_TYPE_ERROR = 'shape must be either a list of ints or an int'
OUPTUT_LENGTH_ZERO = 'output_size cannot be greater than the shapes'
OUPTUT_LENGTH_TOO_LARGE = 'output_size cannot be greater than the shapes'
OUTPUT_LENGTH_MISMATCH = 'The number of outputs does not match the expected number'
SHAPE_MISMATCH = 'Shapes do not match'

def shape_test(*shapes, output_size=1):
    '''
    shapes ([int]): last `output_size` values are expected output sizes
            the others are the input values
    output_size (int): number of elements of `shapes` to treat as the output.
                       must be greater than 0.
    '''
    if output_size == 0:
        raise ValueError(OUTPUT_LENGTH_MISMATCH)

    if output_size >= len(shapes):
        raise ValueError(OUPTUT_LENGTH_TOO_LARGE)

    def decorator(cls):
        class TestClass(cls):
            def __shape_helper(self, shape, f):
                if type(shape) == int:
                    return f([shape])
                elif type(shape) == list:
                    for value in shape:
                        if type(value) != int:
                            raise TypeError(SHAPE_TYPE_ERROR)
                    return f(shape)
                else:
                    raise TypeError(SHAPE_TYPE_ERROR)

            def __random_tensor_from(self, shape):
                return self.__shape_helper(shape, torch.rand)

            def __shape_from(self, shape):
                return self.__shape_helper(shape, torch.Size)

            def test(self):
                input_tensors = [self.__random_tensor_from(shape) for shape in shapes[:-output_size]]
                expected_output_shapes = [self.__shape_from(shape) for shape in shapes[-output_size:]]

                actual_outputs = self.forward(*input_tensors)

                if len(expected_output_shapes) > 1 and len(actual_outputs) != len(expected_output_shapes):
                    raise ValueError(OUTPUT_LENGTH_MISMATCH)

                if type(actual_outputs) == tuple or type(actual_outputs) == list:
                    for actual_output, expected_output_shape in zip(actual_outputs, expected_output_shapes):
                        if actual_output.shape != expected_output_shape:
                            raise ValueError(SHAPE_MISMATCH)
                else:
                    if actual_outputs.shape != expected_output_shapes[0]:
                        raise ValueError(SHAPE_MISMATCH)

            def forward(self, *x):
                return super().forward(*x)

        TestClass().test()

        return cls

    return decorator
