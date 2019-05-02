import codecs
import re
import os
import sys
import math
import pickle
import nltk
from collections import defaultdict

# Add project into python path
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__)))
DATA_PATH = os.path.join(ROOT_PATH, 'data')
WIKI_600_PATH = os.path.join(DATA_PATH, 'wiki-600.txt')
WIKI_26000_PATH = os.path.join(DATA_PATH, 'wiki-26000.txt')
STOP_WORDS = set(nltk.corpus.stopwords.words('english'))
STEMMER = nltk.stem.PorterStemmer()


class SearchEngine(object):
    def __init__(self, wiki_large=False):
        self._articles = None
        self._snippets = None  # Article snippet for display purpose
        self._word_freq = None
        self._file_path = WIKI_26000_PATH if wiki_large else WIKI_600_PATH
        self._wiki_size = 26000 if wiki_large else 600
        self.select_wiki_data()

    def select_wiki_data(self, wiki_large=False, snippet_size=200) -> None:
        """Processes the selected wiki dataset as reference source for search query.

        Articles titles are formatted as follows: = Albert Einstein = .
        If it has two or more = signs on both sides, then it's a subheading.
        The regex looks for article titles, and splits the text file.

        Args:
            wiki_large[boolean] Use large 25000+ Wikipedia articles for search query else
                use smaller 600 Wikipedia articles dataset.
            snippet_size[int] The number of characters to display as summary text.
        """
        self._file_path = WIKI_26000_PATH if wiki_large else WIKI_600_PATH
        self._wiki_size = 26000 if wiki_large else 600
        print('Processing %i articles...' % self._wiki_size)
        text = codecs.open(self._file_path, encoding='utf-8').read()
        starts = [match.span()[0] for match in re.finditer('\n = [^=]', text)]

        # Itemize articles into a list and generate snippets
        self._articles = list()
        for i, start in enumerate(starts):
            end = starts[i + 1] if i + 1 < len(starts) else len(text)
            self._articles.append(text[start:end])
        self._snippets = [' '.join(article[:snippet_size].split()) for article in self._articles]

        # Update word frequency indexing
        self._generate_word_freq()

    @staticmethod
    def _get_tokens(article) -> list:
        """Returns list of tokens for words in the article."""
        tokens = nltk.word_tokenize(article.lower())
        return tokens

    def _index(self, article_id, article, stop_words=True):
        """Perform word frequency indexing for specified article.

        Args:
            article_id[int] Article index in list of articles.
            article[str] String literal representing artcile.
            stop_words[bool] If True, stop words are excluded
        """
        tokens = nltk.word_tokenize(article.lower())
        for token in tokens:
            # Normalize tokens - stems only
            token_stem = STEMMER.stem(token)
            # Exclude stopwords like 'the', 'if', etc for token list
            if stop_words and token_stem in STOP_WORDS:
                continue
            count = self._word_freq[token_stem].get(article_id, 0)
            self._word_freq[token_stem][article_id] = count + 1

    def _generate_word_freq(self):
        self._word_freq = defaultdict(dict)
        increment = 50 if self._wiki_size == 600 else 1000
        print('Generating updated word frequency index. Processed...', end=' ')
        for i, article in enumerate(self._articles):
            if i and i % increment == 0:
                print(i, end=', ')
            sys.stdout.flush()
            self._index(i, article)

    def tf_idf(self, term, article_id):
        """Calculates TF-IDF for each word in occurrences."""
        # Get stem for term, corpus counts, article id count, and
        # total words in article then calculate tf
        stem = STEMMER.stem(term)
        corpus_counts = self._word_freq[stem]
        article_id_count = corpus_counts[article_id]
        words_in_article = float(len(self._articles[article_id]))
        tf = article_id_count / words_in_article

        # Calculate IDF
        total_docs = float(len(self._articles))
        idf = math.log(total_docs / len(corpus_counts))

        return tf * idf

    def save(self, file_name):
        """Saves wiki articles size, snippets, and word frequency index."""
        obj = {
            'size': self._wiki_size,
            'snippets': self._snippets,
            'word_freq': self._word_freq
        }
        print('Saving...')
        file_path = os.path.join(DATA_PATH, file_name)
        with open(file_path, mode='wb') as ff:
            pickle.dump(obj, file=ff)
        print('Done...saved at: ', file_path)

    def load(self, file_name):
        """Update wiki articles size, snippets, and word frequency index."""
        print('Loading...')
        file_path = os.path.join(DATA_PATH, file_name)
        with open(file_path, mode='rb') as ff:
            obj = pickle.load(file=ff)

        print('Done...loaded object from: ', file_path)

        self._word_freq = obj['word_freq']
        self._wiki_size = obj['size']
        self._snippets = obj['snippets']

    def search(self, query, num_results=10):
        """Perform search on articles and returns top results.

        Args:
            query[str] Terms to search for.
            num_results[int] Max number of articles results to return.
        """
        tokens = self._get_tokens(query)
        scores = defaultdict(float)

        # Calculate article scores for query
        for token in tokens:
            for article_id, count in self._word_freq[token].items():
                scores[article_id] += self.tf_idf(token, article_id)

        # Rank articles by score
        result = [(-1, -1)]  # Initialize with placeholder
        for article_id, score in scores.items():
            idx = 0
            while idx < len(result):
                if score > result[idx][0]:
                    result.insert(idx, (score, article_id))
                    break
                idx += 1
            else:
                result.insert(idx - 1, (score, article_id))

        # Remove (-1, -1) placeholder and return top num_results
        result.pop()
        return result[:num_results]

    def display_results(self, query, results):
        """Display results as snippets of articles."""
        print('You searched for: "%s"' % query)
        print('-' * 100)
        for result in results:
            print(self._snippets[result[1]])
        print('=' * 100)
