# TorchSupport

***Note: this library is a work in progress, subject to change without backwards compatability, and is not an official PyTorch library***

TorchSupport is a minimalistic library to support the common patterns and testing of PyTorch based modules. Currently it supports sanity shape tests to ensure that the modules you write take and output tensors of certain shapes as well as a model builder.

***Testing Use Case:***
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

***Model Builder Use Case:***
Often you want to create a pattern of hyperparameters and programmatically specify how each hyperparameter is connected. This library takes a minimalistic approach to solving this problem that is library agnostic.

An example use case would be as follow:
```
d = {
    'lr': SyncOption([1., 1e-1, 1e-2, 1e-3]),
    'n_layers': RangeOption(5, 25, 5),
    'hyper_parameter2': ProductOption([1, 2]),
    'hyper_parameter2': ProductOption([3, 4]),
    'logger': CopyOption(Logger()),
    'constant':  [1, 2]
}

builder = OptionBuilder(d)
```

This will create a builder which can be iterated on like so:
```
options = iter(builder)
for option in options:
    ...
```

This will iterate over each possible instantiation of the hyperparameters, which is represented as a dictionary. In this case, these would be the different option outputs:
```
# Iteration 1
{
    'lr': 1.,
    'n_layers': 5,
    'hyper_parameter1': 1,
    'hyper_parameter2': 3,
    'logger': Logger(),
    'constant':  [1, 2]
}

# Iteration 2
{
    'lr': 1e-1,
    'n_layers': 10,
    'hyper_parameter1': 2,
    'hyper_parameter2': 3,
    'logger': Logger(),
    'constant':  [1, 2]
}

# Iteration 3
{
    'lr': 1e-2,
    'n_layers': 15,
    'hyper_parameter1': 1,
    'hyper_parameter2': 4,
    'logger': Logger(),
    'constant':  [1, 2]
}

# Iteration 4
{
    'lr': 1e-3,
    'n_layers': 20,
    'hyper_parameter1': 2,
    'hyper_parameter2': 4,
    'logger': Logger(),
    'constant':  [1, 2]
}
```

The purpose of each option is as follows:
- `SyncOption`: Each iteration will returns the next item in the list
- `RangeOption`: Each iteration will return the next item in the range. The range is specified in the same format as the pythone `range` method
- `ProductOption`: Will form a cartesian product will all other `ProductOptions`. This will likely be improved in future versions to allow for seperatecartesian products.
- `CopyOption`: Creates a deep copy of the object on each iteration
- Plain object: Outputs the same value for each iteration

When iterating through these, it is important to note that the iterator will stop when the first of these options has been completely iterated, with the exception of `CopyOption` and a plain object will will continuously be iterated over.
