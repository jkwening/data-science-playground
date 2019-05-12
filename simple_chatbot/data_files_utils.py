import os
import codecs
import csv

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
CORPUS_DIR = os.path.join(ROOT_DIR, 'data', 'cornell_movie_dialogs_corpus')
MOVIE_LINES_PATH = os.path.join(CORPUS_DIR, 'movie_lines.txt')
MOVIE_CONVERSATIONS_PATH = os.path.join(CORPUS_DIR, 'movie_conversations.txt')
FORMATTED_MOVIE_LINES_PATH = os.path.join(CORPUS_DIR, 'formatted_movie_lines.txt')
MOVIE_LINES_FIELDS = ['lineID', 'characterID', 'movieID', 'character', 'text']
MOVIE_CONVOS_FIELDS = ['character1ID', 'character2ID', 'movieID', 'utteranceIDs']


def print_lines(file, num_lines=10) -> None:
    with open(file, 'rb') as f:
        lines = f.readlines()

    for line in lines[:num_lines]:
        print(line)


def load_lines(file, fields: list) -> dict:
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


def load_convos(file, lines, fields) -> list:
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


def extract_sentence_pairs(conversations):
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
