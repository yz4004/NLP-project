from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys

import numpy as np
import tensorflow as tf ### import error fixed by reset configuration -- choose base intepreter ?? venv
import keras

from extract_training_data import FeatureExtractor, State

class Parser(object): 

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor
        
        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):
        state = State(range(1,len(words)))
        state.stack.append(0)    

        while state.buffer:
            # TODO: Write the body of this loop for part 4
            x = self.extractor.get_input_representation(words, pos,state)
            # print(x)  # [ 3.  4.  4. 46.  1.  1.]
            x = np.array(x).reshape((1,6))
            y = self.model.predict(x).reshape(91,)   # (1,91) -> (91,)

            # print(y, y.shape)
            # print(np.argmax(y))
            # print("stack", state.stack)

            # if [] is None:
            #     print("yes")
            # if len([]) == 0:
            #     print("yes")

            if len(state.stack) == 0:
                # stack is none, only shift allowable
                state.shift()
                continue

            if state.stack[-1] == 0:
                # top of stack is <ROOT> ~ 0, no left-arc,
                y[1:46] = 0

            if len(state.buffer) == 1:
                # only one word in buffer, shifting it out is illegal, no shift
                y[0] = 0

            # 通过 y （91，） 的 index 0-90 索引到 对应的操作:

            operation = list(self.extractor.output_labels.keys())[np.argmax(y)]
            # 字典的 items keys values 返回一个特殊的可迭代对象，不能直接索引，先list化 再索引
            # 参考 https://www.codenong.com/19030179/?msclkid=89a8a2f3b19911eca405023a73543fdc
            # print(operation) ('right_arc', 'pobj')

            if operation[0] == 'left_arc':
                state.left_arc(self.extractor.output_labels[operation])
            elif operation[0] == 'right_arc':
                state.right_arc(self.extractor.output_labels[operation])
            else: # operation[0] == 'shift'
                state.shift()

            # print(state)
            # print("#"*100)
            # print()

        result = DependencyStructure()
        for p,c,r in state.deps: 
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))
        return result 
        

if __name__ == "__main__":

    # usage:
    #       python decoder.py data/model.h5 data/dev.conll

    # print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    # print(tf.__version__, tf.__path__)

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'


    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r')
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1)

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file:
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
        
