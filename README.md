# TorchTest

***Note: this library is a work in progress, subject to change without backwards compatability, and is not an official PyTorch library***

TorchTest is a library to support the testing of PyTorch based modules. Currently it supports sanity shape tests to ensure that the modules you write take and output tensors of certain shapes.

Example usage:
```
@shape_test(32, 16)
class Network(nn.Module):
    def __init__(self):                                                                                                                               
        super().__init__()                                                                                                                            
        self.layer = nn.Linear(32, 16)                                                                                                                                                      

    def forward(self, x):                                                                                                                             
        return self.layer(x)  
```

This will eagerly test the `Network` module to ensure that given an input tensor of dimensions `torch.Size([32])`, a tensor of dimension `torch.Size([16])` will be outputted after a call on the `forward` method.

This also supports multiple input and multiple outputs as well as stacking the tests. For example,
```
@shape_test(32, 64, 16, 15, output_size=2)
@shape_test([64, 32], 64, [64, 16], 15, output_size=2)
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(32, 16)
        self.layer2 = nn.Linear(64, 15)

    def forward(self, x, y):
        return self.layer1(x), self.layer2(y)
```
will first test if the `forward` method called with tensors of dimensions `torch.Size([64, 32])` and `torch.Size([64])` will output two tensors of dimension `torch.Size([64, 16])` and `torch.Size([15])` resp. Then it will test to ensure that given tensors of sizes `torch.Size([32])` and `torch.Size([64])` passed to the `forward` method, tensors of sizes `torch.Size([16])` and `torch.Size([15])` will be outputted

Notice the use of the `output_size` parameter. This is used if your `forward` method returns multiple tensors. For example if `output_size` was set to 3 in the previous example then the test would expect only one tensor as the input of size `torch.Size([64, 32])`and three outputted tensors of size `torch.Size([64])`, `torch.Size([64, 16])`, and `torch.Size([15])` in the first test above the class, which would fail. The default value is 1 for `output_size`.

When a test fails, an exception will be thrown indicating what type of error occurred, which can be either a `ValueError` or `TypeError` with a short description explaining why the test failed. This will occur once the class is created.
