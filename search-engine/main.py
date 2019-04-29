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
sys.path.append(ROOT_PATH)
DATA_PATH = os.path.join(ROOT_PATH, 'data')
WIKI_600_PATH = os.path.join(DATA_PATH, 'wiki-600.txt')
STOP_WORDS = set(nltk.corpus.stopwords.words('english'))
STEMMER = nltk.stem.PorterStemmer()

# Load data 'wiki-600' data set to begin
text = codecs.open(WIKI_600_PATH, encoding='utf-8').read()
starts = [match.span()[0] for match in re.finditer('\n = [^=]', text)]

# Get index for each article title
# NOTE: Articles titles are formatted as follows: = Albert Einstein = .
# If it has two or more = signs on both sides, then it's a subheading.
# The regex looks for article titles, and splits the text file.
articles = list()
for i, start in enumerate(starts):
    end = starts[i + 1] if i + 1 < len(starts) else len(text)
    articles.append(text[start:end])

# Then calculate snippets i.e. the first 200 characters for each article
snippets = [' '.join(article[:200].split()) for article in articles]

for snippet in snippets[:20]:
    print(snippet)
else:
    print()  # Add EOF line

# Calculate term frequencies
# Track number of times token appears in articles[article_id]
occurrence = defaultdict(dict)


def get_tokens(article):
    tokens = nltk.word_tokenize(article.lower())
    return tokens


def index(article_id, article, stop_words=True):
    tokens = get_tokens(article)
    for token in tokens:
        # Normalize tokens
        token_stem = STEMMER.stem(token)
        # Exclude stopwords like 'the', 'if', etc for token list
        if stop_words and token_stem in STOP_WORDS:
            continue
        count = occurrence[token_stem].get(article_id, 0)
        occurrence[token_stem][article_id] = count + 1


def verify():
    for i, article in enumerate(articles):
        if i and i % 10 == 0:
            print(i, end=', ')
        sys.stdout.flush()
        index(i, article)


def tf_idf(article_id, term):
    """Calculates TF-IDF for each word in occurrences."""
    # Get stem for term, corpus counts, article id count, and
    # total words in article then calculate tf
    stem = STEMMER.stem(term)
    corpus_counts = occurrence[stem]
    article_id_count = corpus_counts[article_id]
    words_in_article = float(len(articles[article_id]))
    tf = article_id_count / words_in_article

    # Calculate IDF
    total_docs = float(len(articles))
    idf = math.log(total_docs / len(corpus_counts))

    return tf * idf


def save(obj, file_name):
    print('Saving...')
    file_path = os.path.join(DATA_PATH, file_name)
    with open(file_path, mode='wb') as ff:
        pickle.dump(obj, file=ff)
    print('Done...saved at: ', file_path)
    return True


def load(file_name):
    print('Loading...')
    file_path = os.path.join(DATA_PATH, file_name)
    with open(file_path, mode='rb') as ff:
        obj = pickle.load(file=ff)

    print('Done...loaded object from: ', file_path)
    return obj


def search(query, num_results=10):
    # populate occurrence dict if empty
    if not occurrence:
        print('\nGenerating token occurrences...')
        verify()
        print('\t...Done!')

    tokens = get_tokens(query)
    scores = defaultdict(float)

    # Calculate article scores for query
    for token in tokens:
        for article_id, count in occurrence[token].items():
            scores[article_id] += tf_idf(article_id, token)

    # Rank articles by score
    result = [(-1, -1)]
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


def display_results(query, results):
    print('You searched for: "%s"' % query)
    print('-' * 100)
    for result in results:
        print(snippets[result[1]])
    print('=' * 100)
