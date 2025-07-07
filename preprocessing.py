import numpy as np
import math
import re
import time
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds

def main():
    
    # Change the path according to your file system organization
    with open('/home/f20212416/FoDS/parallel-n/IITB.en-hi.en',mode='r',encoding='utf-8') as f: 
          eng = f.read()
    with open('/home/f20212416/FoDS/parallel-n/IITB.en-hi.hi',mode='r',encoding='utf-8') as f:
          hindi = f.read()
    with open("/home/f20212416/FoDS/nonbreaking_prefix.en",
              mode='r',
              encoding='utf-8') as f:
        non_breaking_prefix_en = f.read()
    with open("/home/f20212416/FoDS/nonbreaking_prefix.hi",
              mode='r',
              encoding='utf-8') as f:
        non_breaking_prefix_hi = f.read()

    non_breaking_prefix_en = non_breaking_prefix_en.split("\n")
    non_breaking_prefix_en = [' ' + pref.lower() + '.' for pref in non_breaking_prefix_en]
    non_breaking_prefix_hi = non_breaking_prefix_hi.split("\n")
    non_breaking_prefix_hi = [' ' + pref + '.' for pref in non_breaking_prefix_hi]

    corpus_en = eng
    # Add $$$ after non ending sentence points
    for prefix in non_breaking_prefix_en:
        corpus_en = corpus_en.replace(prefix, prefix + '$$$')
    corpus_en = re.sub(r"\.(?=[0-9]|[a-z]|[A-Z])", ".$$$", corpus_en)
    # Remove $$$ markers
    corpus_en = re.sub(r".$$$", '', corpus_en)
    # Clear multiple spaces
    corpus_en = re.sub(r"  +", " ", corpus_en)
    corpus_en = corpus_en.split('\n')
    corpus_en = [x.lower() for x in corpus_en]

    corpus_hi = hindi
    for prefix in non_breaking_prefix_hi:
        corpus_hi = corpus_hi.replace(prefix, prefix + '$$$')
    corpus_hi = re.sub(r"\.(?=[0-9]|[a-z]|[A-Z])", ".$$$", corpus_hi)
    corpus_hi = re.sub(r".$$$", '', corpus_hi)
    corpus_hi = re.sub(r"  +", " ", corpus_hi)
    corpus_hi = corpus_hi.split('\n')

    vocab_en_path = '/home/f20212416/FoDS/vocab/tokenizer_en' 
    vocab_hi_path = '/home/f20212416/FoDS/vocab/tokenizer_hi'

    tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        corpus_en, target_vocab_size=2**13)
    tokenizer_hi = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        corpus_hi, target_vocab_size=2**13)

    tokenizer_en.save_to_file(vocab_en_path)
    tokenizer_hi.save_to_file(vocab_hi_path)
    
    VOCAB_SIZE_EN = tokenizer_en.vocab_size + 2 
    VOCAB_SIZE_HI = tokenizer_hi.vocab_size + 2 

    inputs = [[VOCAB_SIZE_EN-2] + tokenizer_en.encode(sentence) + [VOCAB_SIZE_EN-1]
              for sentence in corpus_en]
    outputs = [[VOCAB_SIZE_HI-2] + tokenizer_hi.encode(sentence) + [VOCAB_SIZE_HI-1]
               for sentence in corpus_hi]

    MAX_LENGTH = 50
    idx_to_remove = [count for count, sent in enumerate(inputs)
                     if len(sent) > MAX_LENGTH]
    for idx in reversed(idx_to_remove):
        del inputs[idx]
        del outputs[idx]
    idx_to_remove = [count for count, sent in enumerate(outputs)
                     if len(sent) > MAX_LENGTH]
    for idx in reversed(idx_to_remove):
        del inputs[idx]
        del outputs[idx]

    inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs,
                                                           value=0,
                                                           padding='post',
                                                           maxlen=MAX_LENGTH)
    outputs = tf.keras.preprocessing.sequence.pad_sequences(outputs,
                                                            value=0,
                                                            padding='post',
                                                            maxlen=MAX_LENGTH)
    np.save('final_data2/inputs', inputs) 
    np.save('final_data2/outputs', outputs) 
    
if __name__ == "__main__":
    main()
