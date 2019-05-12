import random
import os
import torch
import torch.nn as nn

# Project modules
from simple_chatbot.voc import SOS_TOKEN
from simple_chatbot.model_utils import batch_to_train_data, indices_from_sentence
from simple_chatbot.data_files_utils import SENTENCE_MAX_LENGTH, normalize_str


def maskNLLLoss(inp, target, mask, device):
    """
    Loss function calculates the average negative log likelihood of the
    elements that correspond to a '1' in the mask tensor.

    Calculates loss based on model decoder's output tensor, target tensor,
    and a binary mask tensor describing the padding of the target tensor.

    Returns:
        (loss, num of 1 targets)
    """
    n_total = mask.sum()
    cross_entropy = -torch.log(torch.gather(inp, 1,
                                            target.view(-1, 1)).squeeze(1))
    loss = cross_entropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, n_total.item()


def train(input_variable, lengths, target_variable, mask, max_target_len,
          encoder, decoder, encoder_optimizer, decoder_optimizer,
          batch_size, clip, teacher_forcing_ratio, device):
    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_TOKEN for i in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, n_total = maskNLLLoss(decoder_output, target_variable[t], mask[t], device)
            loss += mask_loss
            print_losses.append(mask_loss.item() * n_total)
            n_totals += n_total
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, n_total = maskNLLLoss(decoder_output, target_variable[t], mask[t], device)
            loss += mask_loss
            print_losses.append(mask_loss.item() * n_total)
            n_totals += n_total

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


def train_iters(model_name, voc, pairs, encoder, decoder, encoder_optimizer,
                decoder_optimizer, embedding, encoder_n_layers,
                decoder_n_layers, save_dir, n_iteration, batch_size, print_every,
                save_every, clip, corpus_name, device, load_filename: (bool, object),
                hidden_size, teacher_forcing_ratio=-1):
    """Trains n_iterations of training given the passed arguments."""

    # Load batches for each iteration
    training_batches = [batch_to_train_data(voc, [random.choice(pairs) for _ in range(batch_size)])
                        for _ in range(n_iteration)]

    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    orig_iters = n_iteration
    if load_filename[0]:
        start_iteration = load_filename[1]['iteration'] + 1
        n_iteration += start_iteration - 1

    # Training loop
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        iter_idx = iteration - 1
        if iteration > orig_iters:
            iter_idx = iteration - orig_iters - 1
        training_batch = training_batches[iter_idx]
        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # Run a training iteration with batch
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, encoder_optimizer, decoder_optimizer,
                     batch_size, clip, teacher_forcing_ratio, device=device)
        print_loss += loss

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print(
                "Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".
                    format(iteration, iteration / n_iteration * 100, print_loss_avg)
            )
            print_loss = 0

        # Save checkpoint
        if iteration % save_every == 0:
            directory = os.path.join(save_dir, model_name, corpus_name,
                                     '{}-{}_{}'.format(encoder_n_layers,
                                                       decoder_n_layers,
                                                       hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))


def evaluate(encoder, decoder, searcher, voc, sentence,
             device, max_length=SENTENCE_MAX_LENGTH):
    # Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indices_from_sentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluate_input(encoder, decoder, searcher, voc):
    input_sentence = ''
    while True:
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = normalize_str(input_sentence)
            # Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")
