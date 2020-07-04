import tensorflow as tf
from urllib.request import urlopen
import numpy as np
import os
import time

BATCH_SIZE = 64
BOOK_FILE_ID = {'0':'dorian_gray',
                '1':'war_peace',
                '2':'don_quixote',
                '3':'karamazov_brothers',
                '4':'metamorphosis'}

GUTENBERG_LINK = {'0':'https://www.gutenberg.org/files/174/174.txt',
                '1':'https://www.gutenberg.org/files/2600/2600-0.txt',
                '2':'https://www.gutenberg.org/files/996/996-0.txt',
                '3':'https://www.gutenberg.org/files/28054/28054-0.txt',
                '4':'https://www.gutenberg.org/files/5200/5200.txt'}


def process_text(id):
	url = GUTENBERG_LINK[id]
	text = urlopen(url).read().decode(encoding='utf-8')
	# length of text is the number of characters in it
	print ('Length of text: {} characters'.format(len(text)))

	# The unique characters in the file
	vocab = sorted(set(text))
	print ('{} unique characters'.format(len(vocab)))

	# Creating a mapping from unique characters to indices
	char2idx = {u:i for i, u in enumerate(vocab)}
	idx2char = np.array(vocab)

	text_as_int = np.array([char2idx[c] for c in text])
	return char2idx, idx2char


#TODO: add books
def generate_text(start_string,book_id):
  file_dir = './models/' + BOOK_FILE_ID[book_id] + '.h5'
  model = tf.keras.models.load_model(file_dir)

  # Get Text vars:
  char2idx, idx2char = process_text(book_id)

  # Number of characters to generate
  num_generate = 1000

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the character returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted character as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))


'''
#==============================================
def main():
  file_dir = './models/' + 'metamorphosis'
  model = tf.keras.models.load_model(file_dir)
  print('LOADING and saving....')
  model.save("metamorphosis.h5")

if __name__ == '__main__':
  main()
'''




