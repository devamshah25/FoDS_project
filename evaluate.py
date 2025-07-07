import numpy as np
import math
import re
import time
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import random
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
## Class definitions for each layer of the transformer
class PositionalEncoding(layers.Layer):

    def __init__(self):
        super(PositionalEncoding, self).__init__()
    
    def get_angles(self, pos, i, d_model):
        angles = 1 / np.power(10000., (2*(i//2)) / np.float32(d_model))
        return pos * angles

    def call(self, inputs):
        seq_length = inputs.shape.as_list()[-2]
        d_model = inputs.shape.as_list()[-1]
        angles = self.get_angles(np.arange(seq_length)[:, np.newaxis],
                                 np.arange(d_model)[np.newaxis, :],
                                 d_model)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        pos_encoding = angles[np.newaxis, ...]
        return inputs + tf.cast(pos_encoding, tf.float32)

def scaled_dot_product_attention(queries, keys, values, mask):
    product = tf.matmul(queries, keys, transpose_b=True)
    
    keys_dim = tf.cast(tf.shape(keys)[-1], tf.float32)
    scaled_product = product / tf.math.sqrt(keys_dim)
    
    if mask is not None:
        scaled_product += (mask * -1e9)
    
    attention = tf.matmul(tf.nn.softmax(scaled_product, axis=-1), values)
    
    return attention

class MultiHeadAttention(layers.Layer):
    
    def __init__(self, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        
    def build(self, input_shape):
        self.d_model = input_shape[-1]
        assert self.d_model % self.num_heads == 0
        
        self.d_proj = self.d_model // self.num_heads
        
        self.query_lin = layers.Dense(units=self.d_model)
        self.key_lin = layers.Dense(units=self.d_model)
        self.value_lin = layers.Dense(units=self.d_model)
        
        self.final_lin = layers.Dense(units=self.d_model)
        
    def split_proj(self, inputs, batch_size): # inputs: (batch_size, seq_length, d_model)
        shape = (batch_size,
                 -1,
                 self.num_heads,
                 self.d_proj)
        splited_inputs = tf.reshape(inputs, shape=shape) # (batch_size, seq_length, num_heads, d_proj)
        return tf.transpose(splited_inputs, perm=[0, 2, 1, 3]) # (batch_size, num_heads, seq_length, d_proj)
    
    def call(self, queries, keys, values, mask):
        batch_size = tf.shape(queries)[0]
        
        queries = self.query_lin(queries)
        keys = self.key_lin(keys)
        values = self.value_lin(values)
        
        queries = self.split_proj(queries, batch_size)
        keys = self.split_proj(keys, batch_size)
        values = self.split_proj(values, batch_size)
        
        attention = scaled_dot_product_attention(queries, keys, values, mask)
        
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        
        concat_attention = tf.reshape(attention,
                                      shape=(batch_size, -1, self.d_model))
        
        outputs = self.final_lin(concat_attention)
        
        return outputs

class EncoderLayer(layers.Layer):
    
    def __init__(self, FFN_units, num_heads, dropout_rate):
        super(EncoderLayer, self).__init__()
        self.FFN_units = FFN_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
    
    def build(self, input_shape):
        self.d_model = input_shape[-1]
        
        self.multi_head_attention = MultiHeadAttention(self.num_heads)
        self.dropout_1 = layers.Dropout(rate=self.dropout_rate)
        self.norm_1 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dense_1 = layers.Dense(units=self.FFN_units, activation="relu")
        self.dense_2 = layers.Dense(units=self.d_model)
        self.dropout_2 = layers.Dropout(rate=self.dropout_rate)
        self.norm_2 = layers.LayerNormalization(epsilon=1e-6)
        
    def call(self, inputs, mask, training):
        attention = self.multi_head_attention(inputs,
                                              inputs,
                                              inputs,
                                              mask)
        attention = self.dropout_1(attention, training=training)
        attention = self.norm_1(attention + inputs)
        
        outputs = self.dense_1(attention)
        outputs = self.dense_2(outputs)
        outputs = self.dropout_2(outputs, training=training)
        outputs = self.norm_2(outputs + attention)
        
        return outputs

class Encoder(layers.Layer):
    
    def __init__(self,
                 num_layers,
                 FFN_units,
                 num_heads,
                 dropout_rate,
                 vocab_size,
                 d_model,
                 name="encoder"):
        super(Encoder, self).__init__(name=name)
        self.num_layers = num_layers
        self.d_model = d_model
        
        self.embedding = layers.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding()
        self.dropout = layers.Dropout(rate=dropout_rate)
        self.enc_layers = [EncoderLayer(FFN_units,
                                        num_heads,
                                        dropout_rate) 
                           for _ in range(num_layers)]
    
    def call(self, inputs, mask, training):
        outputs = self.embedding(inputs)
        outputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        outputs = self.pos_encoding(outputs)
        outputs = self.dropout(outputs, training)
        
        for i in range(self.num_layers):
            outputs = self.enc_layers[i](outputs, mask, training)

        return outputs

class DecoderLayer(layers.Layer):
    
    def __init__(self, FFN_units, num_heads, dropout_rate):
        super(DecoderLayer, self).__init__()
        self.FFN_units = FFN_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
    
    def build(self, input_shape):
        self.d_model = input_shape[-1]
        
        # Self multi head attention
        self.multi_head_attention_1 = MultiHeadAttention(self.num_heads)
        self.dropout_1 = layers.Dropout(rate=self.dropout_rate)
        self.norm_1 = layers.LayerNormalization(epsilon=1e-6)
        
        # Multi head attention combined with encoder output
        self.multi_head_attention_2 = MultiHeadAttention(self.num_heads)
        self.dropout_2 = layers.Dropout(rate=self.dropout_rate)
        self.norm_2 = layers.LayerNormalization(epsilon=1e-6)
        
        # Feed foward
        self.dense_1 = layers.Dense(units=self.FFN_units,
                                    activation="relu")
        self.dense_2 = layers.Dense(units=self.d_model)
        self.dropout_3 = layers.Dropout(rate=self.dropout_rate)
        self.norm_3 = layers.LayerNormalization(epsilon=1e-6)
        
    def call(self, inputs, enc_outputs, mask_1, mask_2, training):
        attention = self.multi_head_attention_1(inputs,
                                                inputs,
                                                inputs,
                                                mask_1)
        attention = self.dropout_1(attention, training)
        attention = self.norm_1(attention + inputs)
        
        attention_2 = self.multi_head_attention_2(attention,
                                                  enc_outputs,
                                                  enc_outputs,
                                                  mask_2)
        attention_2 = self.dropout_2(attention_2, training)
        attention_2 = self.norm_2(attention_2 + attention)
        
        outputs = self.dense_1(attention_2)
        outputs = self.dense_2(outputs)
        outputs = self.dropout_3(outputs, training)
        outputs = self.norm_3(outputs + attention_2)
        
        return outputs

class Decoder(layers.Layer):
    
    def __init__(self,
                 num_layers,
                 FFN_units,
                 num_heads,
                 dropout_rate,
                 vocab_size,
                 d_model,
                 name="decoder"):
        super(Decoder, self).__init__(name=name)
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = layers.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding()
        self.dropout = layers.Dropout(rate=dropout_rate)
        
        self.dec_layers = [DecoderLayer(FFN_units,
                                        num_heads,
                                        dropout_rate) 
                           for i in range(num_layers)]
    
    def call(self, inputs, enc_outputs, mask_1, mask_2, training):
        outputs = self.embedding(inputs)
        outputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        outputs = self.pos_encoding(outputs)
        outputs = self.dropout(outputs, training)
        
        for i in range(self.num_layers):
            outputs = self.dec_layers[i](outputs,
                                         enc_outputs,
                                         mask_1,
                                         mask_2,
                                         training)

        return outputs
     
class Transformer(tf.keras.Model):
    
    def __init__(self,
                 vocab_size_enc,
                 vocab_size_dec,
                 d_model,
                 num_layers,
                 FFN_units,
                 num_heads,
                 dropout_rate,
                 name="transformer"):
        super(Transformer, self).__init__(name=name)
        
        self.encoder = Encoder(num_layers,
                               FFN_units,
                               num_heads,
                               dropout_rate,
                               vocab_size_enc,
                               d_model)
        self.decoder = Decoder(num_layers,
                               FFN_units,
                               num_heads,
                               dropout_rate,
                               vocab_size_dec,
                               d_model)
        self.last_linear = layers.Dense(units=vocab_size_dec, name="lin_ouput")
    
    def create_padding_mask(self, seq):
        mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :]

    def create_look_ahead_mask(self, seq):
        seq_len = tf.shape(seq)[1]
        look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        return look_ahead_mask
    
    def call(self, enc_inputs, dec_inputs, training):
        enc_mask = self.create_padding_mask(enc_inputs)
        dec_mask_1 = tf.maximum(
            self.create_padding_mask(dec_inputs),
            self.create_look_ahead_mask(dec_inputs)
        )
        dec_mask_2 = self.create_padding_mask(enc_inputs)
        
        enc_outputs = self.encoder(enc_inputs, enc_mask, training)
        dec_outputs = self.decoder(dec_inputs,
                                   enc_outputs,
                                   dec_mask_1,
                                   dec_mask_2,
                                   training)
        
        outputs = self.last_linear(dec_outputs)
        
        return outputs

def loss_function(target, pred,loss_object):
        mask = tf.math.logical_not(tf.math.equal(target, 0))
        loss_ = loss_object(target, pred)
        
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        
        return tf.reduce_mean(loss_)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    
        def __init__(self, d_model, warmup_steps=4000):
            super(CustomSchedule, self).__init__()
            
            self.d_model = tf.cast(d_model, tf.float32)
            self.warmup_steps = warmup_steps
        
        def __call__(self, step):
            step = tf.cast(step, tf.float32)
            arg1 = tf.math.rsqrt(step)
            arg2 = step * (self.warmup_steps**-1.5)
            
            return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

# def evaluate(inp_sentence):
#     inp_sentence = \
#         [VOCAB_SIZE_EN-2] + tokenizer_en.encode(inp_sentence) + [VOCAB_SIZE_EN-1]
#     enc_input = tf.expand_dims(inp_sentence, axis=0)
        
#     output = tf.expand_dims([VOCAB_SIZE_HI-2], axis=0)
        
#     for _ in range(MAX_LENGTH):
#         predictions = transformer(enc_input, output, False)    
#         prediction = predictions[:, -1:, :] 
#         predicted_id = tf.cast(tf.argmax(prediction, axis=-1), tf.int32)
            
#         if predicted_id == VOCAB_SIZE_HI-1:
#             return tf.squeeze(output, axis=0)
            
#         output = tf.concat([output, predicted_id], axis=-1)
            
#     return tf.squeeze(output, axis=0)

# def translate(sentence):
#     output = evaluate(sentence.lower()).numpy()
        
#     predicted_sentence = tokenizer_hi.decode(
#             [i for i in output if i < VOCAB_SIZE_HI-2]
#         )
        
#     print("Input: {}".format(sentence))
#     print("Predicted translation: {}".format(predicted_sentence))

def main():

    with open('hin.txt', 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if '\t' in line]

    random.shuffle(lines)
    selected_lines = lines[:10]

    english_sentences = []
    hindi_sentences = []

    for line in selected_lines:
        en, hi = line.split('\t')
        english_sentences.append(en.strip())
        hindi_sentences.append(hi.strip())

    # Write to test files
    with open('test.en', 'w', encoding='utf-8') as f_en, \
         open('test.hi', 'w', encoding='utf-8') as f_hi:
        f_en.write('\n'.join(english_sentences))
        f_hi.write('\n'.join(hindi_sentences))

    # --- Step 2: Load tokenizers ---
    vocab_en_path = '/home/f20212416/FoDS/vocab/tokenizer_en' 
    vocab_hi_path = '/home/f20212416/FoDS/vocab/tokenizer_hi'
    tokenizer_en = tfds.features.text.SubwordTextEncoder.load_from_file(vocab_en_path)
    tokenizer_hi = tfds.features.text.SubwordTextEncoder.load_from_file(vocab_hi_path)

    VOCAB_SIZE_EN = tokenizer_en.vocab_size + 2
    VOCAB_SIZE_HI = tokenizer_hi.vocab_size + 2
    MAX_LENGTH = 50

    # Hyper-parameters
    D_MODEL = 128 # 512
    NUM_LAYERS = 4 # 6
    FFN_UNITS = 512 # 2048
    NUM_HEADS = 8 # 8
    DROPOUT_RATE = 0.1 # 0.1
    transformer = Transformer(vocab_size_enc=VOCAB_SIZE_EN,
                              vocab_size_dec=VOCAB_SIZE_HI,
                              d_model=D_MODEL,
                              num_layers=NUM_LAYERS,
                              FFN_units=FFN_UNITS,
                              num_heads=NUM_HEADS,
                              dropout_rate=DROPOUT_RATE)
    # --- Step 3: Evaluation helper ---
    def evaluate(sentence, transformer):
        sentence = [VOCAB_SIZE_EN - 2] + tokenizer_en.encode(sentence.lower()) + [VOCAB_SIZE_EN - 1]
        encoder_input = tf.expand_dims(sentence, 0)
        decoder_input = tf.expand_dims([VOCAB_SIZE_HI - 2], 0)

        for _ in range(MAX_LENGTH):
            predictions = transformer(encoder_input, decoder_input, training=False)
            prediction = predictions[:, -1:, :]
            predicted_id = tf.cast(tf.argmax(prediction, axis=-1), tf.int32)

            if predicted_id == VOCAB_SIZE_HI - 1:
                break
            decoder_input = tf.concat([decoder_input, predicted_id], axis=-1)

        return tf.squeeze(decoder_input, axis=0)

    # --- Step 4: BLEU scoring loop with logging ---

    checkpoint_path = "/home/f20212416/FoDS/checkpoints2/"

    with open("bleu_results.txt", "w", encoding="utf-8") as result_file:
        for ckpt_num in range(6, 11):
            result_file.flush()
            ckpt = tf.train.Checkpoint(transformer=transformer)
            ckpt.restore(f"{checkpoint_path}ckpt-{ckpt_num}").expect_partial()

            bleu_scores = []
            result_file.write(f"\n=== Checkpoint {ckpt_num} ===\n")
            print(f"\n=== Checkpoint {ckpt_num} ===\n")
            for i, (en, hi) in enumerate(zip(english_sentences, hindi_sentences)):
                pred_tensor = evaluate(en, transformer).numpy()
                pred_sentence = tokenizer_hi.decode([i for i in pred_tensor if i < VOCAB_SIZE_HI - 2])
                reference = [hi.split()]
                candidate = pred_sentence.split()
                score = sentence_bleu(reference, candidate, smoothing_function=SmoothingFunction().method1)
                bleu_scores.append(score)

                # Log a few samples
                if i < 5:  # Log first 5 translations
                    result_file.write(f"\nSample {i + 1}:\n")
                    result_file.write(f"EN: {en}\n")
                    result_file.write(f"HI (ref): {hi}\n")
                    result_file.write(f"HI (pred): {pred_sentence}\n")
                    result_file.write(f"BLEU: {score:.4f}\n")
                    print(f"BLEU: {score:.4f}\n")

            avg_bleu = np.mean(bleu_scores)
            result_file.write(f"\nAverage BLEU Score: {avg_bleu:.4f}\n")
            print(f"\nAverage BLEU Score: {avg_bleu:.4f}\n")


    
if __name__ == "__main__":
    main()

