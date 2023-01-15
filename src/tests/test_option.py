from pytest import raises

from torch_support.option import *

class TestRegularOption:
    def test_apply(self):
        option1 = RegularOption([2, 3])
        option2 = RegularOption([3, 4])

        l = RegularOption.apply(option1, option2)

        for _ in range(10):
            assert next(l) == ([2, 3], [3, 4])

class TestSyncOption:
    def test_apply(self):
        option1 = SyncOption([2, 3])
        option2 = SyncOption([4, 5])

        l = SyncOption.apply(option1, option2)

        assert next(l) == (2, 4)
        assert next(l) == (3, 5)
        with raises(StopIteration):
            next(l)

class TestProductOption:
    def test_apply(self):
        option1 = ProductOption([2, 3])
        option2 = ProductOption([4, 5])

        l = ProductOption.apply(option1, option2)

        assert next(l) == (2, 4)
        assert next(l) == (2, 5)
        assert next(l) == (3, 4)
        assert next(l) == (3, 5)

        with raises(StopIteration):
            next(l)


class TestRangeOption:
    def test_apply(self):
        option1 = RangeOption(3)
        option2 = RangeOption(1, 4)

        l = SyncOption.apply(option1, option2)

        assert next(l) == (0, 1)
        assert next(l) == (1, 2)
        assert next(l) == (2, 3)
        with raises(StopIteration):
            next(l)


class TestCopyOption:
    def test_apply(self):
        option1 = CopyOption(iter(range(3)))

        l = SyncOption.apply(option1)

        a = next(l)[0]
        b = next(l)[0]

        assert next(a) == 0
        assert next(b) == 0 # make sure they are different objects

class TestNamedOption:
    def test_apply(self):
        option1 = NamedOption('a', SyncOption([2, 3]))
        option2 = NamedOption('b', SyncOption([4, 5]))

        l = NamedOption.apply(option1, option2)

        assert next(l) == (('a', 2), ('b', 4))
        assert next(l) == (('a', 3), ('b', 5))
        with raises(StopIteration):
            next(l)



class TestOptionBuilder:
    def test_next(self):
        d = {
            'option1': SyncOption([1, 2, 3]),
            'option2': SyncOption([4, 5, 6]),
            'option3': RegularOption(7)
        }

        builder = OptionBuilder(d)
        i = iter(builder)

        assert next(i) == {
            'option1': 1,
            'option2': 4,
            'option3': 7
        }

        assert next(i) == {
            'option1': 2,
            'option2': 5,
            'option3': 7
        }

        assert next(i) == {
            'option1': 3,
            'option2': 6,
            'option3': 7
        }

        with raises(StopIteration):
            next(i)

    def test_next_with_plain_argument(self):
        d = {
            'option1': SyncOption([1, 2, 3]),
            'option2': SyncOption([4, 5, 6]),
            'option3': 7
        }

        builder = OptionBuilder(d)
        i = iter(builder)

        assert next(i) == {
            'option1': 1,
            'option2': 4,
            'option3': 7
        }

        assert next(i) == {
            'option1': 2,
            'option2': 5,
            'option3': 7
        }

        assert next(i) == {
            'option1': 3,
            'option2': 6,
            'option3': 7
        }

        with raises(StopIteration):
            next(i)
