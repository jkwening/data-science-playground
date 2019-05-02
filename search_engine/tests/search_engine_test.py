import unittest
import math
from search_engine.search_engine import SearchEngine


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        # reset occurrence for each unit test
        self._engine = SearchEngine()

    def test_get_tokens(self):
        expected = 34
        tokens = self._engine._get_tokens(self._engine._snippets[0])
        self.assertEqual(expected, len(tokens),
                         msg=f'Expected {expected}\t Got {len(tokens)} tokens')

    def test_index(self):
        expected = {
            '=': {0: 2, 1: 2}, 'valkyria': {0: 4}, 'chronicl': {0: 3},
            'iii': {0: 2}, 'senjō': {0: 1}, 'no': {0: 1}, '3': {0: 2},
            ':': {0: 2}, 'unrecord': {0: 1}, '(': {0: 1}, 'japanes': {0: 1},
            '戦場のヴァルキュリア3': {0: 1}, ',': {0: 2, 1: 2}, 'lit': {0: 1},
            '.': {0: 1}, 'of': {0: 1, 1: 2}, 'the': {0: 1, 1: 3},
            'battlefield': {0: 1}, ')': {0: 1}, 'commonli': {0: 1},
            'refer': {0: 1}, 'to': {0: 1}, 'as': {0: 1, 1: 1},
            'outsid': {0: 1}, 'tower': {1: 2}, 'build': {1: 4},
            'littl': {1: 3}, 'rock': {1: 2}, 'arsen': {1: 3}, 'also': {1: 1},
            'known': {1: 1}, 'u.s.': {1: 1}, 'is': {1: 1}, 'a': {1: 1},
            'locat': {1: 1}, 'in': {1: 2}, 'macarthur': {1: 1},
            'park': {1: 1}, 'downtown': {1: 1}, 'roc': {1: 1}
        }
        snippets = self._engine._snippets[0:2]
        for i, snippet in enumerate(snippets):
            self._engine._index(i, snippet, stop_words=False)

        print(self._engine._word_freq, '\n')
        for k1, v1 in expected.items():
            self.assertTrue(k1 in self._engine._word_freq.keys(),
                            msg=f'Expected {k1} in {self._engine._word_freq.keys()}')
            for k2, v2 in v1.items():
                self.assertEqual(v2, self._engine._word_freq[k1][k2],
                                 msg=f'For {k1} expected {k2, v2} in {self._engine._word_freq[k1]}')

    def test_verify(self):
        # Expected output: {300: 1, 84: 5, 294: 1}
        # That is, articles[300] has token einstein 1 times, articles[84] has
        # token einstein 5 times, and articles[294] has token einstein 1 times.
        expected = {300: 1, 84: 5, 294: 1}
        self._engine._generate_word_freq()
        result = self._engine._word_freq['einstein']

        for id in expected:
            self.assertTrue(id in result,
                            msg=f'Expected {id} in {result.keys()}')
            self.assertEqual(expected[id], result[id],
                             msg=f'Expected {id, expected[id]} in {result}')

    def test_tf_idf(self):
        sample = {
            'valkyria': {0: 4}, 'the': {0: 1, 1: 3}
        }
        self._engine._word_freq = sample

        # Case 'valkyria'
        tf_idf = self._engine.tf_idf(article_id=0, term='valkyria')
        tf = 4.0 / len(self._engine._articles[0])
        idf = math.log(float(len(self._engine._articles)) / 1)
        self.assertEqual(tf * idf, tf_idf)

        # Case 'the'
        tf_idf = self._engine.tf_idf(article_id=1, term='the')
        tf = 3.0 / len(self._engine._articles[1])
        idf = math.log(float(len(self._engine._articles)) / 2)
        self.assertEqual(tf * idf, tf_idf)

    def test_search(self):
        # Case 1
        results = self._engine.search('einstein')
        self.assertEqual(3, len(results))

        # Case 2
        results = self._engine.search('obama')
        self.assertEqual(8, len(results))

        # Case 1
        results = self._engine.search('india')
        self.assertEqual(10, len(results))

    def test_print(self):
        queries = ['obama', 'einstein', 'physics', 'india', 'director']
        for query in queries:
            results = self._engine.search(query)
            self._engine.display_results(query, results)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
