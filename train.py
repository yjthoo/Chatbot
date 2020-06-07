# import libraries
import numpy as np
import re
import time
import os

import tensorflow as tf
from sklearn.model_selection import train_test_split

import preprocessing
from seq2seqModel import Encoder, Decoder, BahdanauAttention

import zipfile

# unzip file
with zipfile.ZipFile("cornell_movie_dialogs_corpus.zip", 'r') as zip_ref:
    zip_ref.extractall()

# import the datasets
lines = open('cornell movie-dialogs corpus/movie_lines.txt', encoding='utf-8', errors = "ignore").read().split("\n")
conversations = open('cornell movie-dialogs corpus/movie_conversations.txt', encoding='utf-8', errors = "ignore").read().split("\n")

# Creating a dictionary that maps each line and its id
id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]

# Creating a list of all of the conversations
conversations_ids = []
for conversation in conversations[:-1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
    conversations_ids.append(_conversation.split(','))

# getting separately the questions and the answers
rawQuestions = []
rawAnswers = []

for conversation in conversations_ids:
    for i in range(len(conversation) - 1):
        rawQuestions.append(id2line[conversation[i]])
        rawAnswers.append(id2line[conversation[i+1]])

questions = []
answers = []

for question in rawQuestions:
    questions.append(preprocessing.clean_text(question))

for answer in rawAnswers:
    answers.append(preprocessing.clean_text(answer))

# Filtering out the questions and answers that are too short or too long
short_questions = []
short_answers = []
i = 0
for question in questions:
    if 2 <= len(question.split()) <= 25:
        short_questions.append(question)
        short_answers.append(answers[i])
    i += 1
questions = []
answers = []
i = 0
for answer in short_answers:
    if 2 <= len(answer.split()) <= 25:
        answers.append(answer)
        questions.append(short_questions[i])
    i += 1

# sort the questions and answers by the length of the questions (speeds up the training by reducing padding)
sorted_questions = []
sorted_answers = []

for questLen in range(1, 25 + 1):
    for idx, quest in enumerate(questions):
        if len(quest) == questLen:
            sorted_questions.append(preprocessing.preprocess_sentence(questions[idx]))
            sorted_answers.append(preprocessing.preprocess_sentence(answers[idx]))

input_tensor, target_tensor, tokenizer = preprocessing.tokenize(sorted_questions, sorted_answers)
max_input_length, max_target_length = input_tensor.shape[1], input_tensor.shape[1]

input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

# setting the hyperparameters
batch_size = 64
lstm_units = 512
num_layers = 3
encoder_embedding_size = 512
decoder_embedding_size = 512
steps_per_epoch = len(input_tensor_train)//batch_size

vocab_size = len(tokenizer.word_index)+1

# dropout rate of 50% for hidden units
keep_probability = 0.5

# prepare datasets
dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(len(input_tensor_train))
dataset = dataset.batch(batch_size, drop_remainder=True)

dataset_val = tf.data.Dataset.from_tensor_slices((input_tensor_val, target_tensor_val)).shuffle(len(input_tensor_val))
dataset_val = dataset_val.batch(batch_size, drop_remainder=True)

input_shape = input_tensor.shape

initial_learning_rate = 0.01

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100,
    decay_rate=0.96,
    staircase=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred, input_shape, sequence_length=25):
    #https://www.tensorflow.org/addons/api_docs/python/tfa/seq2seq/sequence_loss
    #return tfa.seq2seq.sequence_loss(pred, real, tf.ones([input_shape[0], sequence_length]))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

encoder = Encoder(vocab_size, encoder_embedding_size, lstm_units, batch_size, keep_probability)
decoder = Decoder(vocab_size, decoder_embedding_size, lstm_units, batch_size)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

@tf.function
def train_step(inputs, targets, encoder_hidden):

    loss = 0

    with tf.GradientTape() as tape:
        encoder_output, encoder_hidden = encoder(inputs, encoder_hidden)

        decoder_hidden = encoder_hidden

        decoder_input = tf.expand_dims([tokenizer.word_index['<start>']] * batch_size, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targets.shape[1]):
            # passing enc_output to the decoder
            predictions, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_output)

            #print(targets[:, t].shape, predictions.shape, input_shape)
            loss += loss_function(targets[:, t], predictions, input_shape)

            # using teacher forcing
            decoder_input = tf.expand_dims(targets[:, t], 1)

    batch_loss = (loss / int(targets.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


def evaluate(inputs, targets, encoder_hidden):

    loss = 0
    encoder_output, encoder_hidden = encoder(inputs, encoder_hidden)
    decoder_hidden = encoder_hidden

    decoder_input = tf.expand_dims([tokenizer.word_index['<start>']] * batch_size, 1)

    # Teacher forcing - feeding the target as the next input
    for t in range(1, targets.shape[1]):
        # passing enc_output to the decoder
        predictions, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_output)
        loss += loss_function(targets[:, t], predictions, input_shape)

        # using teacher forcing
        decoder_input = tf.expand_dims(targets[:, t], 1)

    batch_loss = (loss / int(targets.shape[1]))
    return batch_loss


EPOCHS = 60
batch_training_loss_check = 100
batch_validation_loss_check = steps_per_epoch // 2 - 1

# elements for early stopping
list_validation_loss_error = []
early_stopping_check = 0
early_stopping_stop = 100

for epoch in range(EPOCHS):

    enc_hidden = encoder.initialize_hidden_state()
    total_train_loss = 0

    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        start = time.time()
        batch_train_loss = train_step(inp, targ, enc_hidden)
        total_train_loss += batch_train_loss
        end = time.time()
        batch_time = end - start

        if batch % batch_training_loss_check == 0:
            print('Epoch {}, Batch {}, Training Loss {:.3f}, Training Time on {} batches: {:.2f} s'.format(epoch + 1,
                                                   batch, batch_train_loss.numpy(), batch_training_loss_check,
                                                   batch_time))

        if batch % batch_validation_loss_check == 0 and batch > 0:
            total_valid_loss = 0
            start_val_time = time.time()

            for (batch_val, (inp_val, targ_val)) in enumerate(dataset_val.take(len(input_tensor_val)//batch_size)):
                total_valid_loss += evaluate(inp_val, targ_val, enc_hidden)

            ending_val_time = time.time()
            val_time = ending_val_time - start_val_time

            average_validation_loss = total_valid_loss / (len(input_tensor_val) / batch_size)
            print("Avg Validation Loss Error: {:>6.3f}, Validation Time: {:.2f} s".format(average_validation_loss, val_time))

            # early stopping
            list_validation_loss_error.append(average_validation_loss)

            if average_validation_loss <= min(list_validation_loss_error):
                early_stopping_check = 0
                checkpoint.save(file_prefix = checkpoint_prefix)
            else:
                early_stopping_check += 1

            if early_stopping_check == early_stopping_stop:
                break

    print('Epoch {}, Training Loss {:.3f}, Avg Validation Loss {:.3f}\n'.format(epoch + 1,
                                                                                total_train_loss / steps_per_epoch,
                                                                                average_validation_loss))

    if early_stopping_check == early_stopping_stop:
        break
