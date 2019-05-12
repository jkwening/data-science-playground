# Default word tokens
PAD_TOKEN = 0  # Used for padding short sentences
SOS_TOKEN = 1  # Start-of-sentence token
EOS_TOKEN = 2  # End-of-sentence token


class Voc(object):
    """
    Voc class keeps a mapping from words to indices, a reverse mapping of
    indices to words, a count of each word, and a total word count.
    """
    def __init__(self, name):
        self._name = name
        self._trimmed = False
        self._word2index = dict()
        self._word2count = dict()
        self._index2word = {
            PAD_TOKEN: 'PAD',
            SOS_TOKEN: 'SOS',
            EOS_TOKEN: 'EOS'
        }
        self._num_words = 3  # Count SOS, EOS, PAD

    def get_num_words(self) -> int:
        return self._num_words

    def get_word2index(self):
        return self._word2index

    def add_word(self, word):
        """Add a word to the vocabulary."""
        if word not in self._word2index:
            self._word2index[word] = self._num_words
            self._word2count[word] = 1
            self._index2word[self._num_words] = word
            self._num_words += 1
        else:
            self._word2count[word] += 1

    def add_sentence(self, sentence):
        """Add all words in a sentence to vocabulary."""
        for word in sentence.split():
            self.add_word(word)

    def trim(self, min_count):
        """
        Trim infrequently seen words by removing words below a
        certain count threshold.
        """
        if self._trimmed:  # Skip if vocabulary already trimmed
            return

        self._trimmed = True
        keep_words = list()

        for k, v in self._word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('\nKeep words {} / {} = {:.4f}'.format(
            len(keep_words), len(self._word2index),
            len(keep_words) / len(self._word2index)
        ))

        # Re-initialize dictionaries
        self._word2index = dict()
        self._word2count = dict()
        self._index2word = {
            PAD_TOKEN: 'PAD',
            SOS_TOKEN: 'SOS',
            EOS_TOKEN: 'EOS'
        }
        self._num_words = 3  # Count default tokens

        # Update vocabulary with kept words
        for word in keep_words:
            self.add_word(word)
