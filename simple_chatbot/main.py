import os
import sys
import pickle

# Add project to sys.path if not available
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import torch
import torch.nn as nn
import torch.optim as optim

# Project modules
from simple_chatbot.models import EncoderRNN, DecoderRNN
from simple_chatbot.train_eval_utils import train_iters
from simple_chatbot.data_files_utils import SAVE_DIR, CORPUS_NAME

# Load saved voc and pairs objects
with open('data/save/voc_05122019', mode='rb') as f:
    voc = pickle.load(f)
with open('data/save/pairs_05122019', mode='rb') as f:
    pairs = pickle.load(f)


# Configure models
model_name = 'cb_model'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64
voc_num_words = voc.get_num_words()
corpus_name = CORPUS_NAME

# Set checkpoint to load from; set to None if starting from scratch
# loadFilename = None
checkpoint_iter = 8000
loadFilename = os.path.join(SAVE_DIR, model_name, corpus_name,
                           '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
                           '{}_checkpoint.tar'.format(checkpoint_iter))


# Load model if a loadFilename is provided
if loadFilename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']


# Use gpu if available
USE_CUDA = torch.cuda.is_available()
device = 'cuda' if USE_CUDA else 'cpu'

print(f'Building encoder and decoder on {device}...')
# Initialize word embeddings
embedding = nn.Embedding(voc_num_words, hidden_size)
if loadFilename:
    embedding.load_state_dict(embedding_sd)

# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = DecoderRNN(attn_model, embedding, hidden_size, voc_num_words, decoder_n_layers, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)

# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')

# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 20000
print_every = 1
save_every = 500

# Ensure dropout layers are in train mode
encoder.train()
decoder.train()

# Initialize optimizers
print('Building optimizers ...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
if loadFilename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

# Run training iterations
print("Starting Training!")
train_iters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
           embedding, encoder_n_layers, decoder_n_layers, SAVE_DIR, n_iteration, batch_size,
           print_every, save_every, clip, corpus_name, device, (True, checkpoint), hidden_size)
