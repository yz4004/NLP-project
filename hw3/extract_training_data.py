from conll_reader import DependencyStructure, conll_reader
from collections import defaultdict
import copy
import sys
import keras
import numpy as np
from keras.utils.np_utils  import to_categorical


class State(object):
    def __init__(self, sentence=[]):
        self.stack = []
        self.buffer = []
        if sentence:
            self.buffer = list(reversed(sentence))
        self.deps = set()

    def shift(self):
        self.stack.append(self.buffer.pop())
        # shift add nothing to dependency set

    def left_arc(self, label):
        self.deps.add((self.buffer[-1], self.stack.pop(), label))

    def right_arc(self, label):
        parent = self.stack.pop()
        self.deps.add((parent, self.buffer.pop(), label))
        self.buffer.append(parent)

    def __repr__(self):
        return "{},{},{}".format(self.stack, self.buffer, self.deps)


def apply_sequence(seq, sentence):
    state = State(sentence)
    for rel, label in seq:
        if rel == "shift":
            state.shift()
        elif rel == "left_arc":
            state.left_arc(label)
        elif rel == "right_arc":
            state.right_arc(label)

    return state.deps


class RootDummy(object):
    def __init__(self):
        self.head = None
        self.id = 0
        self.deprel = None

    def __repr__(self):
        return "<ROOT>"


def get_training_instances(dep_structure):
    deprels = dep_structure.deprels

    sorted_nodes = [k for k, v in sorted(deprels.items())]  # cmp???
    state = State(sorted_nodes)
    state.stack.append(0)

    childcount = defaultdict(int)
    for ident, node in deprels.items():
        childcount[node.head] += 1
        # count how many children does a word (head) has


    seq = []
    while state.buffer:
        if not state.stack: # if stack is null, do shift
            seq.append((copy.deepcopy(state), ("shift", None)))
            state.shift()
            continue
        if state.stack[-1] == 0:
            stackword = RootDummy()
        else:
            stackword = deprels[state.stack[-1]]
        bufferword = deprels[state.buffer[-1]]
        if stackword.head == bufferword.id:
            childcount[bufferword.id] -= 1
            seq.append((copy.deepcopy(state), ("left_arc", stackword.deprel)))
            state.left_arc(stackword.deprel)
        elif bufferword.head == stackword.id and childcount[bufferword.id] == 0:
            childcount[stackword.id] -= 1
            seq.append((copy.deepcopy(state), ("right_arc", bufferword.deprel)))
            state.right_arc(bufferword.deprel)
        else:
            seq.append((copy.deepcopy(state), ("shift", None)))
            state.shift()
    return seq


dep_relations = ['tmod', 'vmod', 'csubjpass', 'rcmod', 'ccomp', 'poss', 'parataxis', 'appos', 'dep', 'iobj', 'pobj',
                 'mwe', 'quantmod', 'acomp', 'number', 'csubj', 'root', 'auxpass', 'prep', 'mark', 'expl', 'cc',
                 'npadvmod', 'prt', 'nsubj', 'advmod', 'conj', 'advcl', 'punct', 'aux', 'pcomp', 'discourse',
                 'nsubjpass', 'predet', 'cop', 'possessive', 'nn', 'xcomp', 'preconj', 'num', 'amod', 'dobj', 'neg',
                 'dt', 'det']


class FeatureExtractor(object):

    def __init__(self, word_vocab_file, pos_vocab_file):
        self.word_vocab = self.read_vocab(word_vocab_file)  # training corpus
        self.pos_vocab = self.read_vocab(pos_vocab_file)    # pos tag of training corpus
        self.output_labels = self.make_output_labels()      # dependency relation

    def make_output_labels(self):
        labels = []
        labels.append(('shift', None))

        for rel in dep_relations:
            labels.append(("left_arc", rel))
            labels.append(("right_arc", rel))
        return dict((label, index) for (index, label) in enumerate(labels))

    def read_vocab(self, vocab_file):
        vocab = {}
        for line in vocab_file:
            word, index_s = line.strip().split()
            index = int(index_s)
            vocab[word] = index
        return vocab

    def get_input_representation(self, words, pos, state):
        # TODO: Write this method for Part 2

        stack = state.stack
        buffer = state.buffer

        # print()
        # print("#####################"*2)      00000000000000
        # print("#####################"*2)
        # print(stack)
        # print(buffer)

        '''
        for example, first sentence in dev.conll: 
        stack: [0]
        buffer: [30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 
        19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        '''
        result = np.zeros(6)

        # padding <NULL> 4
        l = len(stack)
        while l < 3:
            result[l] = 4
            l += 1

        l = len(buffer)
        while l < 3:
            result[l + 3] = 4
            l += 1

        for i in range(1, len(stack)+1 ):
            # words[stack[i]], pos[stack[i]]
            # i-th in stack (bottom up), an integer -- word index in sentence
            # words[stack[i]]  is the word, pos[stack[i]] is its pos tag

            # print(words[stack[- i]], end=" ")  111111111111

            if stack[-i] == 0:
                result[i-1] = 3
            elif pos[stack[-i]] == 'CD':  # 判断数字 不需要用 word.is_number() 我们有训练集的 pos tag、 直接检查 是不是 CD 就行了
                result[i-1] = 0
            elif pos[stack[-i]] == 'NNP':
                result[i-1] = 1
            else:
                try:
                    result[i-1] = self.word_vocab[words[stack[-i]]]
                except KeyError:
                    # keyError: None
                    result[i-1] = 4
                    '''
                    当 word_vocab 中没有这个词时，test 里出现了 train 里没有的词，这时标记为 none 4
                    用 error catch 来替换
                    '''

            if i == 3:
                break

        # print("-----", end= " ")       2222222222222

        for i in range(1,len(buffer)+1 ):

            # print(words[buffer[- i]], end = " ")      3333333333333
            if buffer[-i] == 0:
                result[i+2] = 3               # i +3 -1
            elif pos[buffer[-i]] == 'CD':
                result[i+2] = 0
            elif pos[buffer[-i]] == 'NNP':
                result[i+2] = 1
            else:
                try:
                    result[i+2] = self.pos_vocab[ pos[buffer[-i]] ]
                except KeyError:
                    result[i+2] = 0

            if i == 3:
                break

        # print("\n", result)           4444444444444

        return result

    def get_output_representation(self, output_pair):
        # TODO: Write this method for Part 2

        label = self.output_labels[output_pair]

        return to_categorical(label, 91)

    # to_categorical 接受整数encoded 向量，同时指定 所有可以的编码范围，即最大的 vocab size 1 - 91 然后生成 one_hot 向量

    # https://www.geeksforgeeks.org/python-keras-keras-utils-to_categorical/#:~:text=Keras%20provides%20numpy%20utility%20library%2C%20which%20provides%20functions,to%20the%20number%20of%20categories%20in%20the%20data.?msclkid=fadc3e38af2b11ecaa3256b5af554ea0

'''
array([[6],
       [9],
       [9],
       ...,
       [9],
       [1],
       [1]], dtype=uint8)
       
       变成
       
[[0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 1. 0.]
 [0. 0. 0. ... 0. 1. 0.]
 ...
 [0. 0. 0. ... 0. 0. 0.]
 [0. 1. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 1. 0. 0.]]
'''


def get_training_matrices(extractor, in_file):
    inputs = []
    outputs = []
    count = 0
    for dtree in conll_reader(in_file):  # dtree is a sentence DependencyStructure
        words = dtree.words()
        pos = dtree.pos()
        for state, output_pair in get_training_instances(dtree):
            inputs.append(extractor.get_input_representation(words, pos, state))
            outputs.append(extractor.get_output_representation(output_pair))
            # print(outputs[-1])

            # print(inputs)
            # print(outputs)
            # print("##"*50)
        if count % 100 == 0:
            sys.stdout.write(".")
            sys.stdout.flush() # u can try what will happen without flush stdout buffer
        count += 1
    sys.stdout.write("\n")
    return np.vstack(inputs), np.vstack(outputs)

# https://www.geeksforgeeks.org/numpy-vstack-in-python/?msclkid=aeea4b76ae6411ec8227399ded5b751a

# vstack 将 a list of （n,）vectors 纵向堆叠


if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE, 'r')
        pos_vocab_f = open(POS_VOCAB_FILE, 'r')
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1)

    with open(sys.argv[1], 'r') as in_file:

        extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
        print("Starting feature extraction... (each . represents 100 sentences)")
        inputs, outputs = get_training_matrices(extractor, in_file)
        print("Writing output...")
        np.save(sys.argv[2], inputs)
        np.save(sys.argv[3], outputs)
