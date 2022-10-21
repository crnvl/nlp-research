import os
import keras_nlp
import tensorflow as tf
from tensorflow import keras

# Dataset Settings
BATCH_SIZE = 64
SEQ_LEN = 128
MIN_TRAINING_SEQ_LEN = 450

# Model Settings
EMBED_DIM = 256
FEED_FORWARD_DIM = 256
NUM_HEADS = 3
NUM_LAYERS = 2
VOCAB_SIZE = 5000

# Training Settings
EPOCHS = 30

# Inference
NUM_TOKENS_TO_GENERATE = 128

keras.utils.get_file(origin="https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip", extract=True, )

dir = os.path.expanduser("~/.keras/datasets/wikitext-103-raw/")

vocab_file = keras.utils.get_file(
    origin="https://storage.googleapis.com/tensorflow/keras-nlp/examples/bert/bert_vocab_uncased.txt",
)

# Load dataset and filter out short lines
raw_train_ds = (
    tf.data.TextLineDataset(dir + "wiki.train.raw")
    .filter(lambda x: tf.strings.length(x) > MIN_TRAINING_SEQ_LEN)
    .batch(BATCH_SIZE)
    .shuffle(buffer_size=256)
)

# Load validation dataset and filter out short lines too
raw_val_ds = (
    tf.data.TextLineDataset(dir + "wiki.valid.raw")
    .filter(lambda x: tf.strings.length(x) > MIN_TRAINING_SEQ_LEN)
    .batch(BATCH_SIZE)
)

vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
    raw_train_ds,
    vocabulary_size=VOCAB_SIZE,
    lowercase=True,
    reserved_tokens=["[PAD]", "[UNK]", "[BOS]"],
)

tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=vocab,
    sequence_length=SEQ_LEN,
    lowercase=True,
)

start_packer = keras_nlp.layers.StartEndPacker(
    sequence_length=SEQ_LEN,
    start_value=tokenizer.token_to_id("[BOS]"),
)

def preprocess(inputs):
    outputs = tokenizer(inputs)
    features = start_packer(outputs)
    labels = outputs
    return features, labels

train_ds = raw_train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
val_ds = raw_val_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

inputs = keras.layers.Input(shape=(None,), dtype=tf.int32)

embedding_layer = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=tokenizer.vocabulary_size(),
    sequence_length=SEQ_LEN,
    embedding_dim=EMBED_DIM,
    mask_zero=True,
)

x = embedding_layer(inputs)

for _ in range(NUM_LAYERS):
    decoder_layer = keras_nlp.layers.TransformerDecoder(
        num_heads=NUM_HEADS,
        intermediate_dim=FEED_FORWARD_DIM
    )
    x = decoder_layer(x)

outputs = keras.layers.Dense(VOCAB_SIZE)(x)

model = keras.Model(inputs, outputs)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
perplexity = keras_nlp.metrics.Perplexity(from_logits=True, mask_token_id=0)

model.compile(optimizer="adam", loss=loss_fn, metrics=[perplexity])

model.summary()

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)

model.save('./trained_wiki_' + str(EPOCHS))

prompt_tokens = tf.convert_to_tensor([tokenizer.token_to_id("[BOS]")])

def token_logits_fn(inputs):
    cur_len = inputs.shape[1]
    output = model(inputs)
    return output[:, cur_len - 1, :]  # return next token logits

output_tokens = keras_nlp.utils.top_p_search(
    token_logits_fn,
    prompt_tokens,
    max_length=NUM_TOKENS_TO_GENERATE,
    p=0.5,
    from_logits=True,
)
txt = tokenizer.detokenize(output_tokens)
print(f"Top-P search generated text: \n{txt}\n")