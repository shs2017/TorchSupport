"""
Utility module for testing pytorch based module (i.e. torch.nn.Module)

Decorators:
    shape_test

Note that this is dependent on the pytorch library.
"""

import torch

SHAPE_TYPE_ERROR = "shape must be either a list of ints or an int"
OUPTUT_LENGTH_ZERO = "output_size cannot be greater than the shapes"
OUPTUT_LENGTH_TOO_LARGE = "output_size cannot be greater than the shapes"
OUTPUT_LENGTH_MISMATCH = "The number of outputs does not match the expected number"
SHAPE_MISMATCH = "Shapes do not match"


def shape_test(*shapes, output_size=1):
    """A decorator to eagerly run a sanity shape test, to ensure that the output(s)
    is the correct shape of a module.

    Params:
          shapes ([int]): last `output_size` values are expected output sizes
                          the others are the input values
          output_size (int): number of elements of `shapes` to treat as the output.
                             must be greater than 0.
    """
    if output_size == 0:
        raise ValueError(OUTPUT_LENGTH_MISMATCH)

    if output_size >= len(shapes):
        raise ValueError(OUPTUT_LENGTH_TOO_LARGE)

    def decorator(cls):
        """decorator method"""

        class TestClass(cls):  # pylint: disable=too-few-public-methods
            """This is an internal class which is used
            in order to eagerly run the shape tests"""

            def __shape_helper(self, shape, func):
                if isinstance(shape, int):
                    return func([shape])

                if isinstance(shape, list):
                    for value in shape:
                        if not isinstance(value, int):
                            raise TypeError(SHAPE_TYPE_ERROR)

                    return func(shape)

                raise TypeError(SHAPE_TYPE_ERROR)

            def __random_tensor_from(self, shape):
                return self.__shape_helper(shape, torch.rand)

            def __shape_from(self, shape):
                return self.__shape_helper(shape, torch.Size)

            def test(self):
                """Runs the sanity test on the shapes of the module"""
                input_tensors = [
                    self.__random_tensor_from(shape) for shape in shapes[:-output_size]
                ]

                expected_output_shapes = [
                    self.__shape_from(shape) for shape in shapes[-output_size:]
                ]

                actual_outputs = self.forward(*input_tensors)

                if len(expected_output_shapes) > 1 and len(actual_outputs) != len(
                    expected_output_shapes
                ):
                    raise ValueError(OUTPUT_LENGTH_MISMATCH)

                if isinstance(actual_outputs, (tuple, list)):
                    for actual_output, expected_output_shape in zip(
                        actual_outputs, expected_output_shapes
                    ):
                        if actual_output.shape != expected_output_shape:
                            raise ValueError(SHAPE_MISMATCH)
                else:
                    if actual_outputs.shape != expected_output_shapes[0]:
                        raise ValueError(SHAPE_MISMATCH)

        TestClass().test()

        return cls

    return decorator
