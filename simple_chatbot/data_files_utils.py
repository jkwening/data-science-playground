import os
import codecs
import csv
import unicodedata
import re
import pickle
from datetime import datetime

# Project modules
from simple_chatbot.voc import Voc

TIMESTAMP = datetime.now().strftime('%m%d%Y')
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
SAVE_DIR = os.path.join(ROOT_DIR, 'data', 'save')
CORPUS_DIR = os.path.join(ROOT_DIR, 'data', 'cornell_movie_dialogs_corpus')
MOVIE_LINES_PATH = os.path.join(CORPUS_DIR, 'movie_lines.txt')
MOVIE_CONVERSATIONS_PATH = os.path.join(CORPUS_DIR, 'movie_conversations.txt')
FORMATTED_MOVIE_LINES_PATH = os.path.join(CORPUS_DIR, 'formatted_movie_lines.txt')
MOVIE_LINES_FIELDS = ['lineID', 'characterID', 'movieID', 'character', 'text']
MOVIE_CONVOS_FIELDS = ['character1ID', 'character2ID', 'movieID', 'utteranceIDs']
SENTENCE_MAX_LENGTH = 10


def print_lines(file: str, num_lines=10) -> None:
    with open(file, 'rb') as f:
        lines = f.readlines()

    for line in lines[:num_lines]:
        print(line)


def load_lines(file: str, fields: list) -> dict:
    """Parse each line of file into dictionary of fields."""
    lines = dict()
    with open(file, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(' +++$+++ ')
            line_obj = dict()
            for i, field in enumerate(fields):
                line_obj[field] = values[i]

            lines[line_obj['lineID']] = line_obj
    return lines


def load_convos(file: str, lines: list, fields: list) -> list:
    """
    Groups fields of lines from load_lines() into conversations based o
     move_conversations.txt
    """
    convos = list()
    with open(file, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(' +++$+++ ')
            conv_obj = dict()
            for i, field in enumerate(fields):
                conv_obj[field] = values[i]

            # Convert str to list (conv_obj['utteranceIds'] == ['L598485', 'L598486', ...]"
            line_ids = eval(conv_obj['utteranceIDs'])

            # Reassemble lines
            conv_obj['lines'] = list()
            for line_id in line_ids:
                conv_obj['lines'].append(lines[line_id])

            convos.append(conv_obj)
    return convos


def extract_sentence_pairs(conversations: list) -> list:
    """Extracts pairs of sentences from conversations."""
    pairs = list()
    for conv in conversations:
        for i in range(len(conv['lines']) - 1):
            input_line = conv['lines'][i]['text'].strip()
            target = conv['lines'][i + 1]['text'].strip()

            if input_line and target:
                pairs.append([input_line, target])
    return pairs


def format_movie_lines(save_file_as='formatted_movie_lines.txt'):
    # Un-escape the delimiter
    delimiter = '\t'
    delimiter = str(codecs.decode(delimiter, 'unicode_escape'))

    # Load lines and process conversations
    print('\nProcessing corpus...')
    lines = load_lines(file=MOVIE_LINES_PATH, fields=MOVIE_LINES_FIELDS)
    print('\nLoading conversations...')
    convos = load_convos(file=MOVIE_CONVERSATIONS_PATH, lines=lines,
                         fields=MOVIE_CONVOS_FIELDS)

    # Write new csv file
    print('\nWriting newly formatted file...')
    with open(save_file_as, 'w', encoding='utf-8') as data:
        writer = csv.writer(data, delimiter=delimiter, lineterminator='\n')
        for pair in extract_sentence_pairs(convos):
            writer.writerow(pair)

    # Print a sample of lines
    print('\nSample lines from file:')
    print_lines(save_file_as)


def unicode_to_ascii(s: str) -> str:
    """Turn a Unicode string to plain ASCII.

    Thanks to: https://stackoverflow.com/a/518232/2809427
    """
    return ''.join(
        [c for c in unicodedata.normalize('NFD', s)
         if unicodedata.category(c) != 'Mn']
    )


def normalize_str(s: str) -> str:
    """
    Normalize string by converting to lowercase, trimming, and removing
    non-letter characters.
    """
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r'([.!?])', r' \1', s)
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    s = re.sub('\s+', r' ', s).strip()
    return s


def read_vocs(formatted_data_file_path, corpus_name):
    """Read query/response pairs.

    Returns:
        voc, pairs: tuple Voc object and pairs of normalized lines.
    """
    print('\nReading lines...')

    # Read the file and split into lines
    lines = open(formatted_data_file_path, encoding='utf-8').read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalize_str(s) for s in l.split('\t')] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs


def filter_pair(p: str) -> bool:
    """
    Returns True iff both sentences in a pair 'p' are under
    the SENTENCE_MAX_LENGTH threshold.
    """
    return len(p[0].split()) < SENTENCE_MAX_LENGTH and \
           len(p[1].split()) < SENTENCE_MAX_LENGTH


def filter_pairs(pairs: list) -> list:
    """Filter pairs using filter_pair() condition."""
    return [pair for pair in pairs if filter_pair(pair)]


def trim_rare_words(voc, pairs, min_count) -> list:
    """Trim words under min_count threshold using 'voc.trim'."""
    voc.trim(min_count)

    # Filter out pairs with trimmed words
    keep_pairs = list()
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True

        # Check input and output sentences
        word2index = voc.get_word2index()
        for word in input_sentence.split():
            if word not in word2index:
                keep_input = False
                break
        for word in output_sentence.split():
            if word not in word2index:
                keep_output = False
                break

        # Update keep_pairs tracker
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print('\tTrimmed from {} pairs to {}, {:.4f} of total'.format(
        len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)
    ))
    return keep_pairs


def assemble_voc_pairs(corpus_name: str, formatted_data_file_path: str,
                       voc_trim: (bool, int), save=True) -> tuple:
    """
    Process formatted movie lines data file and save voc obj
    and pairs by default.

    Args:
        corpus_name: Name of the corpus.
        formatted_data_file_path: The path for the formatted movie lines txt.
        voc_trim (bool, min_count): Trim works used under the min_count from
          the voc. Example: (True, 3) trims words with word counts less than 3
          from voc.
        save: If True save voc object and pairs list as pickle file.

    Returns:
        voc, pairs: tuple Voc object and pairs of normalized lines.
    """
    print('\nStart preparing training data...')
    voc, pairs = read_vocs(formatted_data_file_path, corpus_name)
    print('\tRead {!s} sentence pairs'.format(len(pairs)))
    pairs = filter_pairs(pairs)
    print('\tFiltered to {!s} sentence pairs'.format(len(pairs)))

    print('\nCounting vocab words...')
    for pair in pairs:
        voc.add_sentence(pair[0])
        voc.add_sentence(pair[1])
    print('\tCounted vocab words:', voc.get_num_words())

    # Trim rare words
    if voc_trim[0]:
        pairs = trim_rare_words(voc, pairs, min_count=voc_trim[1])

    # Print some pairs to validate
    print('\nPairs:')
    for pair in pairs[:10]:
        print(pair)

    if save:
        print('\nSaving voc object via pickle...')
        with open(os.path.join(SAVE_DIR, f'voc_{TIMESTAMP}'), mode='wb') as f:
            pickle.dump(voc, f)

        print('\nSaving pairs list via pickle...')
        with open(os.path.join(SAVE_DIR, f'pairs_{TIMESTAMP}'), mode='wb') as f:
            pickle.dump(pairs, f)

    return voc, pairs


temp_voc, temp_pairs = assemble_voc_pairs(corpus_name='Cornell Movie Dialogues',
                                          formatted_data_file_path=FORMATTED_MOVIE_LINES_PATH,
                                          voc_trim=(True, 3), save=True)
