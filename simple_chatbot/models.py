import torch
import torch.nn as nn
import torch.nn.functional as F

# Project modules
from simple_chatbot.voc import SOS_TOKEN


class EncoderRNN(nn.Module):
    """
    RNN model that encodes a variable-length input sequence to a
    fixed-length context vector.
    """
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU
        # Note: the input_size and hidden_size params are both set to
        # 'hidden_size' because our input size is a word embedding with
        # number of features == hidden_size.
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size,
                          num_layers=n_layers,
                          dropout=(0 if n_layers == 1 else dropout),
                          bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        """
        Args:
            input_seq: Batch of input sentences; shape=(max_length, batch_size)
            input_lengths: List of sentence lengths corresponding to each sentence
              in the batch; shape=(batch_size)
            hidden: The hidden state; shape=(n_layers*num_directions,
              batch_size, hidden_size)

        Returns:
            outputs: The output features from the last hidden layer of the GRU;
              shape=(max_length, batch_size, hidden_size)
            hidden: The updated hidden state from teh GRU;
              shape=(n_layers*num_directions, batch_size, hidden_size)
        """
        # Convert word indices to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(input=embedded,
                                                   lengths=input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(sequence=outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + \
            outputs[:, :, self.hidden_size:]
        return outputs, hidden


class Attn(nn.Module):
    """Luong attention layer."""
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'genera', 'concat']:
            raise ValueError(self.method, 'is not an appropriate method.')

        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(in_features=hidden_size,
                                  out_features=hidden_size)
        if self.method == 'concat':
            self.attn = nn.Linear(in_features=hidden_size * 2,
                                  out_features=hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1),
                                      encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        else:
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch)size dimensions
        attn_energies = attn_energies.t()

        # Return softmax normalized probability scores (with added dimension)
        return F.softmax(input=attn_energies, dim=1).unsqueeze(1)


class DecoderRNN(nn.Module):
    """
    RNN model seq2seq decoder implementation that takes an input word and
    the context vector, and returns a guess for the next word in the
    sequence and a hidden state to use in the next iteration.

    This implementation utilized Luong's attention layer to add attention
    mechanism to the seq2seq mode.
    """
    def __init__(self, attn_model, embedding, hidden_size, output_size,
                 n_layers=1, dropout=0.1):
        super(DecoderRNN, self).__init__()

        # Module variables
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size,
                          num_layers=n_layers,
                          dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(in_features=hidden_size * 2,
                                out_features=hidden_size)
        self.out = nn.Linear(in_features=hidden_size,
                             out_features=output_size)
        self.attn = Attn(method=attn_model, hidden_size=hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        """Forward pass, one step (word) at a time."""
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new
        # 'weighted sum' context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # PRedict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        return output, hidden


class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, dtype=torch.long) * SOS_TOKEN
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], dtype=torch.long)
        all_scores = torch.zeros([0])
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores
