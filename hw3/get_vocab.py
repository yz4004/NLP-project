import sys
from conll_reader import conll_reader
from collections import defaultdict

def get_vocabularies(conll_reader):
    word_set = defaultdict(int)
    pos_set = set()
    for dtree in conll_reader:
        for ident, node in dtree.deprels.items():
            if node.pos != "CD" and node.pos!="NNP":
                word_set[node.word.lower()] += 1
            pos_set.add(node.pos)

    word_set = set(x for x in word_set if word_set[x] > 1)
    # print(pos_set)
    # {'VBN', 'POS', 'CD', 'WP$', 'DT', 'VB', 'PRP$', 'RP', '``', 'RB', 'SYM', 'IN', 'VBD', '-LRB-', 'JJS', 'JJR', 'NNS',
    #  ':', 'CC', 'RBS', 'LS', 'PDT', 'VBZ', 'RBR', "''", ',', 'VBP', 'WP', 'NN', '.', '$', 'NNPS', 'VBG', 'EX', '-RRB-',
    #  'WRB', 'PRP', 'TO', 'WDT', 'UH', '#', 'FW', 'JJ', 'MD', 'NNP'}

    # print('UNK' in pos_set) -- False

    word_list = ["<CD>","<NNP>","<UNK>","<ROOT>","<NULL>"] + list(word_set)
    pos_list =  ["<UNK>","<ROOT>","<NULL>"] + list(pos_set)

    return word_list, pos_list 

if __name__ == "__main__":
    with open(sys.argv[1],'r') as in_file, open(sys.argv[2],'w') as word_file, open(sys.argv[3],'w') as pos_file:
        word_list, pos_list = get_vocabularies(conll_reader(in_file))
        print("Writing word indices...")
        for index, word in enumerate(word_list): 
            word_file.write("{}\t{}\n".format(word, index))
        print("Writing POS indices...")
        for index, pos in enumerate(pos_list): 
            pos_file.write("{}\t{}\n".format(pos, index))
        
        


