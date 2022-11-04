from extract_training_data import FeatureExtractor
import sys
import numpy as np
import tensorflow as tf
import keras
from keras import Sequential
from keras.layers import Flatten, Embedding, Dense, Input

def build_model(word_types, pos_types, outputs):
    # TODO: Write this function for part 3
    model = Sequential()
    #model.add(...)
    # model.add(I)
    model.add(Embedding(word_types, 32, input_length=6)) # N*6 -> N*32*6
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(outputs, activation='softmax'))
    # model.compile(keras.optimizers.Adam(lr=0.01), loss="categorical_crossentropy") version issue
    model.compile(tf.keras.optimizers.Adam(learning_rate=0.01), loss="categorical_crossentropy")
    return model

'''
https://keras.io/api/layers/core_layers/embedding/

输入 一个 N*d 的整数编码的矩阵 N 为 sample size d 为每个训练样本的 size

其中 每个整数 的 取值范围 为 1 - 15152, 15152 为 我们文本库 vocabulary 的 size

embedding 想法是 将 15152 的 one hot  映射成 一个 k 的 hidden layer 再 映射回 15152 大小 的 概率

这里 我们的 k 是 32， 而这个 32 就是一个 原先 one_hot 的 简洁表示 

所以 对于每一个 词都影射一个 32 长度 新表示，原先为 6 的 x_train 行有 6 个词，现在 变成一个 32*6 的 矩阵

整个 x_train N*6 -> N * 32 * 6

'''

if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    print("Compiling model.")
    # words.vocab 15152     pos.vocab 47     (left_arc, ...), (right_arc, ...) (shift, ...) 91
    model = build_model(len(extractor.word_vocab), len(extractor.pos_vocab), len(extractor.output_labels))
    inputs = np.load(sys.argv[1])
    outputs = np.load(sys.argv[2])
    print("Done loading data.")
   
    # Now train the model
    model.fit(inputs, outputs, epochs=5, batch_size=100)
    
    model.save(sys.argv[3])
