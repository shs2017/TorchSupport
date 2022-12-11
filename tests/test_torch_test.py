from torch import nn
from pytest import raises

from torch_test.utils import shape_test

class TestTorchTest:
    def test_single(self):
        @shape_test(32, 16)
        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = nn.Linear(32, 16)

            def forward(self, x):
                return self.l1(x)

    def test_multiple(self):
        @shape_test(32, 64, 16, 15, output_size=2)
        @shape_test([64, 32], 64, [64, 16], 15, output_size=2)
        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = nn.Linear(32, 16)
                self.l2 = nn.Linear(64, 15)

            def forward(self, x, y):
                return self.l1(x), self.l2(y)

    def test_failed(self):
        with raises(ValueError):
            @shape_test(32, 64, 16, 15, output_size=2)
            class Network(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.l1 = nn.Linear(32, 16)
                    self.l2 = nn.Linear(64, 16)

                def forward(self, x, y):
                    return self.l1(x), self.l2(y)

    def test_invalid_type_float(self):
        with raises(TypeError):
            @shape_test(32, 64, 3.2, 16, output_size=2)
            class Network(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.l1 = nn.Linear(32, 16)
                    self.l2 = nn.Linear(64, 16)

                def forward(self, x, y):
                    return self.l1(x), self.l2(y)

    def test_invalid_type_empty(self):
        with raises(TypeError):
            @shape_test(32, 64, None, 16, output_size=2)
            class Network(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.l1 = nn.Linear(32, 16)
                    self.l2 = nn.Linear(64, 16)

                def forward(self, x, y):
                    return self.l1(x), self.l2(y)

    def test_invalid_type_in_list(self):
        with raises(TypeError):
            @shape_test(32, 64, 16, 15, output_size=2)
            @shape_test([64, 32], 64, [64, 3.2], 15, output_size=2)
            class Network(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.l1 = nn.Linear(32, 16)
                    self.l2 = nn.Linear(64, 15)

                def forward(self, x, y):
                    return self.l1(x), self.l2(y)

    def test_invalid_output_size(self):
        with raises(ValueError):
            @shape_test(32, 64, 16, 16, output_size=10)
            class Network(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.l1 = nn.Linear(32, 16)
                    self.l2 = nn.Linear(64, 16)

                def forward(self, x, y):
                    return self.l1(x), self.l2(y)

    def test_multiple_one_fails(self):
        with raises(ValueError):
            @shape_test(32, 64, 15, 15, output_size=2)
            @shape_test([64, 32], 64, [64, 16], 15, output_size=2)
            class Network(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.l1 = nn.Linear(32, 16)
                    self.l2 = nn.Linear(64, 15)

                def forward(self, x, y):
                    return self.l1(x), self.l2(y)

    def test_multiple_two_fail(self):
        with raises(TypeError):
            @shape_test(32, 64, 15, 15, output_size=2)
            @shape_test([64, 32], 64, [64, 3.2], 15, output_size=2)
            class Network(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.l1 = nn.Linear(32, 16)
                    self.l2 = nn.Linear(64, 15)

                def forward(self, x, y):
                    return self.l1(x), self.l2(y)
