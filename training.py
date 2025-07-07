import numpy as np
import math
import re
import time
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds

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

def evaluate(inp_sentence):
    inp_sentence = \
        [VOCAB_SIZE_EN-2] + tokenizer_en.encode(inp_sentence) + [VOCAB_SIZE_EN-1]
    enc_input = tf.expand_dims(inp_sentence, axis=0)
        
    output = tf.expand_dims([VOCAB_SIZE_HI-2], axis=0)
        
    for _ in range(MAX_LENGTH):
        predictions = transformer(enc_input, output, False)    
        prediction = predictions[:, -1:, :] 
        predicted_id = tf.cast(tf.argmax(prediction, axis=-1), tf.int32)
            
        if predicted_id == VOCAB_SIZE_HI-1:
            return tf.squeeze(output, axis=0)
            
        output = tf.concat([output, predicted_id], axis=-1)
            
    return tf.squeeze(output, axis=0)

def translate(sentence):
    output = evaluate(sentence.lower()).numpy()
        
    predicted_sentence = tokenizer_hi.decode(
            [i for i in output if i < VOCAB_SIZE_HI-2]
        )
        
    print("Input: {}".format(sentence))
    print("Predicted translation: {}".format(predicted_sentence))

def main():

    ## Below code is for preprocessing the data. The commented part only needs to be run once.
    # with open('/home/f20212416/FoDS/parallel-n/IITB.en-hi.en',mode='r',encoding='utf-8') as f:
    #     eng = f.read()
    # with open('/home/f20212416/FoDS/parallel-n/IITB.en-hi.hi',mode='r',encoding='utf-8') as f:
    #     hindi = f.read()
    # with open("/home/f20212416/FoDS/nonbreaking_prefix.en",
    #         mode='r',
    #         encoding='utf-8') as f:
    #     non_breaking_prefix_en = f.read()
    # with open("/home/f20212416/FoDS/nonbreaking_prefix.hi",
    #         mode='r',
    #         encoding='utf-8') as f:
    #     non_breaking_prefix_hi = f.read()

    # non_breaking_prefix_en = non_breaking_prefix_en.split("\n")
    # non_breaking_prefix_en = [' ' + pref.lower() + '.' for pref in non_breaking_prefix_en]
    # non_breaking_prefix_hi = non_breaking_prefix_hi.split("\n")
    # non_breaking_prefix_hi = [' ' + pref + '.' for pref in non_breaking_prefix_hi]

    # corpus_en = eng
    # # Add $$$ after non ending sentence points
    # for prefix in non_breaking_prefix_en:
    #     corpus_en = corpus_en.replace(prefix, prefix + '$$$')
    # corpus_en = re.sub(r"\.(?=[0-9]|[a-z]|[A-Z])", ".$$$", corpus_en)
    # # Remove $$$ markers
    # corpus_en = re.sub(r".$$$", '', corpus_en)
    # # Clear multiple spaces
    # corpus_en = re.sub(r"  +", " ", corpus_en)
    # corpus_en = corpus_en.split('\n')
    # corpus_en = [x.lower() for x in corpus_en]
        
    # corpus_hi = hindi
    # for prefix in non_breaking_prefix_hi:
    #     corpus_hi = corpus_hi.replace(prefix, prefix + '$$$')
    # corpus_hi = re.sub(r"\.(?=[0-9]|[a-z]|[A-Z])", ".$$$", corpus_hi)
    # corpus_hi = re.sub(r".$$$", '', corpus_hi)
    # corpus_hi = re.sub(r"  +", " ", corpus_hi)
    # corpus_hi = corpus_hi.split('\n')

    vocab_en_path = '/home/f20212416/FoDS/vocab/tokenizer_en' 
    vocab_hi_path = '/home/f20212416/FoDS/vocab/tokenizer_hi'

    # tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    #     corpus_en, target_vocab_size=2**13)
    # tokenizer_hi = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    #     corpus_hi, target_vocab_size=2**13)

    # tokenizer_en.save_to_file(vocab_en_path)
    # tokenizer_hi.save_to_file(vocab_hi_path)

    MAX_LENGTH = 50
    tokenizer_en = tfds.features.text.SubwordTextEncoder.load_from_file(vocab_en_path)
    tokenizer_hi = tfds.features.text.SubwordTextEncoder.load_from_file(vocab_hi_path)

    VOCAB_SIZE_EN = tokenizer_en.vocab_size + 2 
    VOCAB_SIZE_HI = tokenizer_hi.vocab_size + 2 

    # inputs = [[VOCAB_SIZE_EN-2] + tokenizer_en.encode(sentence) + [VOCAB_SIZE_EN-1]
    #         for sentence in corpus_en]
    # outputs = [[VOCAB_SIZE_HI-2] + tokenizer_hi.encode(sentence) + [VOCAB_SIZE_HI-1]
    #         for sentence in corpus_hi]

    # MAX_LENGTH = 50
    # idx_to_remove = [count for count, sent in enumerate(inputs)
    #                 if len(sent) > MAX_LENGTH]
    # for idx in reversed(idx_to_remove):
    #     del inputs[idx]
    #     del outputs[idx]
    # idx_to_remove = [count for count, sent in enumerate(outputs)
    #                 if len(sent) > MAX_LENGTH]
    # for idx in reversed(idx_to_remove):
    #     del inputs[idx]
    #     del outputs[idx]

    # inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs,
    #                                                     value=0,
    #                                                     padding='post',
    #                                                     maxlen=MAX_LENGTH)
    # outputs = tf.keras.preprocessing.sequence.pad_sequences(outputs,
    #                                                         value=0,
    #                                                         padding='post',
    #                                                         maxlen=MAX_LENGTH)

    # np.save('final_data/inputs', inputs) 
    # np.save('final_data/outputs', outputs) 

    inputs = np.load('final_data/inputs.npy') 
    outputs = np.load('final_data/outputs.npy')

    BATCH_SIZE = 64
    BUFFER_SIZE = 20000

    dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs))

    dataset = dataset.cache()
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    tf.keras.backend.clear_session()

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

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                            reduction="none")


    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

    leaning_rate = CustomSchedule(D_MODEL)
    #leaning_rate = 0.001
    print(leaning_rate)
    optimizer = tf.keras.optimizers.Adam(leaning_rate,
                                            beta_1=0.9,
                                            beta_2=0.98,
                                            epsilon=1e-9)

    checkpoint_path = "/home/f20212416/FoDS/checkpoints2/"

    ckpt = tf.train.Checkpoint(transformer=transformer,
                            optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Latest checkpoint restored!!")
    
    EPOCHS = 10
    for epoch in range(EPOCHS):
        print("Start of epoch {}".format(epoch+1))
        start = time.time()
        
        train_loss.reset_states()
        train_accuracy.reset_states()
        
        for (batch, (enc_inputs, targets)) in enumerate(dataset):
            dec_inputs = targets[:, :-1]
            dec_outputs_real = targets[:, 1:]
            with tf.GradientTape() as tape:
                predictions = transformer(enc_inputs, dec_inputs, True)
                loss = loss_function(dec_outputs_real, predictions,loss_object)
            
            gradients = tape.gradient(loss, transformer.trainable_variables)
            optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
            
            train_loss(loss)
            train_accuracy(dec_outputs_real, predictions)
            
            if batch % 50 == 0:
                print("Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}".format(
                    epoch+1, batch, train_loss.result(), train_accuracy.result()))
                
        ckpt_save_path = ckpt_manager.save()
        print("Saving checkpoint for epoch {} at {}".format(epoch+1,
                                                            ckpt_save_path))
        print("Time taken for 1 epoch: {} secs\n".format(time.time() - start))



if __name__ == "__main__":
    main()
    

    


                