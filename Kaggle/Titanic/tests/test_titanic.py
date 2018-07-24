import unittest
from .. import titanic


class MyTestCase(unittest.TestCase):
    def test_(self):


    def test_load_training_set(self):
        result = titanic.load_training_set()
        self.assertTrue(result, 'Null pandas data object returned')


if __name__ == '__main__':
    unittest.main()
