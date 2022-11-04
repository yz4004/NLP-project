"""
COMS W4705 - Natural Language Processing
Homework 2 - Parsing with Context Free Grammars 
Yassine Benajiba
"""
import math
import sys
from collections import defaultdict
import itertools
from grammar import Pcfg


### Use the following two functions to check the format of your data structures in part 3 ###
def check_table_format(table):
    """
    Return true if the backpointer table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict):
        sys.stderr.write("Backpointer table is not a dict.\n")
        return False
    for split in table:
        if not isinstance(split, tuple) and len(split) == 2 and \
                isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the backpointer table must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of backpointer table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str):
                sys.stderr.write(
                    "Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            bps = table[split][nt]
            if isinstance(bps, str):  # Leaf nodes may be strings
                continue
            if not isinstance(bps, tuple):
                sys.stderr.write(
                    "Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Incorrect type: {}\n".format(
                        bps))
                return False
            if len(bps) != 2:
                sys.stderr.write(
                    "Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Found more than two backpointers: {}\n".format(
                        bps))
                return False
            for bp in bps:
                if not isinstance(bp, tuple) or len(bp) != 3:
                    sys.stderr.write(
                        "Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has length != 3.\n".format(
                            bp))
                    return False
                if not (isinstance(bp[0], str) and isinstance(bp[1], int) and isinstance(bp[2], int)):
                    print(bp)
                    sys.stderr.write(
                        "Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has incorrect type.\n".format(
                            bp))
                    return False
    return True


def check_probs_format(table):
    """
    Return true if the probability table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict):
        sys.stderr.write("Probability table is not a dict.\n")
        return False
    for split in table:
        if not isinstance(split, tuple) and len(split) == 2 and isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the probability must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of probability table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str):
                sys.stderr.write(
                    "Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            prob = table[split][nt]
            if not isinstance(prob, float):
                sys.stderr.write(
                    "Values of the inner dictionary (for each span and nonterminal) must be a float.{}\n".format(prob))
                return False
            if prob > 0:
                sys.stderr.write("Log probability may not be > 0.  {}\n".format(prob))
                return False
    return True


class CkyParser(object):
    """
    A CKY parser.
    """

    def __init__(self, grammar):
        """
        Initialize a new parser instance from a grammar. 
        """
        self.grammar = grammar

    def is_in_language(self, tokens):
        """
        Membership checking. Parse the input tokens and return True if 
        the sentence is in the language described by the grammar. Otherwise
        return False
        """
        # TODO, part 2

        n = len(tokens)

        # initialization -- fill out (i,i+1) diagonal
        table = defaultdict(set)
        # s = set()

        rhs = self.grammar.rhs_to_rules
        for i in range(n):
            # nt = rhs.get((tokens[i],))[-1][0]   # rhs.get() returns a list of tuple, tuple[0] is nt
            # print(nt)
            # # this nt is a word nt, unique
            # table[(i, i + 1)] = {nt} # {nt: (0,) }  # {nt: ((tokens[i],-1,-1), (tokens[i],-1,-1)) }

            for grammar_tuple in rhs.get((tokens[i],)):
                nt = grammar_tuple[0]
                table[(i, i + 1)].add(nt)

        for length in range(2, n + 1):

            for i in range(0, n - length + 1):

                j = i + length

                for k in range(i + 1, j - 1 + 1):

                    #
                    print(i, k, table.get((i, k)))
                    for B in table.get((i, k)):

                        for C in table.get((k, j)):

                            if table.get((i, j)) is None:
                                print(i, j)
                                table[(i, j)] = {"##NONE"}
                                print(table[(i, j)])
                            table[(i, j)].add("##NONE")

                            if rhs.get((B, C)) is not None:
                                # ('ADVP', 'NP')  [('FRAG', ('ADVP', 'NP'), 0.0344827586207), ('NPBAR', ('ADVP', 'NP'), 0.00428265524625), ('VPBAR', ('ADVP', 'NP'), 0.00602409638554)]
                                # table.get(i,j).add(rhs.get((B, C))[] )
                                for grammar_tuple in rhs.get((B, C)):
                                    table.get((i, j)).add(
                                        grammar_tuple[0])  # ('NP', 'APART') [('ADVP', ('NP', 'APART'), 0.153846153846)]

                                for grammar_tuple in rhs.get((B, C)):
                                    table.get((i, j)).add(
                                        grammar_tuple[0])  # ('NP', 'APART') [('ADVP', ('NP', 'APART'), 0.153846153846)]

        for i in range(0, len(toks) + 1):

            for j in range(i + 1, len(toks) + 1):
                print(i, j, " ", table[(i, j)], end="  ####  ")
            print()

        result = False
        if self.grammar.startsymbol in table.get((0, n)):  # 'TOP'
            result = True
        return result

    def generator(self):
        tuple = ("0", 0, 0)
        return tuple

    def parse_with_backpointers(self, tokens):
        """
        Parse the input tokens and return a parse table and a probability table.
        """
        # TODO, part 3
        table = defaultdict(lambda: defaultdict(lambda: defaultdict(tuple)))
        probs = defaultdict(lambda: defaultdict(lambda: -float("inf")))

        n = len(tokens)

        rhs = self.grammar.rhs_to_rules
        for i in range(n):

            try:
                for grammar_tuple in rhs.get((tokens[i],)):
                    nt = grammar_tuple[0]

                    probs[(i, i + 1)][nt] = math.log(grammar_tuple[-1])

                    table[(i, i + 1)][nt] = grammar_tuple[1][0]

                    # table[(i,i+1)] = {grammar_tuple[0]: grammar_tuple[1][0]}
                    # probs[(i,i+1)] = {grammar_tuple[0]: math.log(grammar_tuple[-1])} # ('be',) [('BE', ('be',), 1.0)]

                    # probs[(i,i+1)] = defaultdict(float) # ('be',) [('BE', ('be',), 1.0)]]

                    # probs[(i, i + 1)] = defaultdict(lambda: -float("inf"))
                    # table[(i, i +1 )] = defaultdict(str)

                    ## print(i,i+1,nt, probs[(i,i+1)][nt],math.log(grammar_tuple[-1]),probs[(i,i+1)][nt] < math.log(grammar_tuple[-1]))

            except:
                continue

        for length in range(2, n + 1):

            for i in range(0, n - length + 1):
                j = i + length

                # table[(i, j)] = defaultdict(tuple)  # table[(i, j)] = defaultdict(lambda:tuple())
                # probs[(i, j)] = defaultdict(float)

                for k in range(i + 1, j - 1 + 1):

                    try:
                        for B in table.get((i, k)):

                            try:
                                for C in table.get((k, j)):
                                    if rhs[(B, C)] is not None:
                                        for A_BC_p in rhs[(B, C)]:
                                            A = A_BC_p[0]
                                            if probs[(i, j)][A] < math.log(A_BC_p[2]) + probs.get((i, k))[B] + \
                                                    probs.get((k, j))[C]:
                                                probs[(i, j)][A] = math.log(A_BC_p[2]) + probs.get((i, k))[B] + \
                                                                   probs.get((k, j))[C]
                                                table[(i, j)][A] = ((B, i, k), (C, k, j))
                            except:
                                continue

                    except:
                        continue

                if table.get((i, j)) is None:
                    table[(i, j)] = defaultdict()
                    probs[(i, j)] = defaultdict()

        return table, probs


def get_tree(chart, i, j, nt):
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.

    initial call (table,    0,len(toks),  "TOP" )

    """
    # TODO: Part 4

    if isinstance(chart.get((i, j))[nt], str):
        return nt, chart.get((i, j))[nt]

    left_tuple, right_tuple = chart[(i, j)][nt]

    return (nt, get_tree(chart, left_tuple[1], left_tuple[2], left_tuple[0]),
            get_tree(chart, right_tuple[1], right_tuple[2], right_tuple[0]))


if __name__ == "__main__":
    with open('atis3.pcfg', 'r') as grammar_file:
        grammar = Pcfg(grammar_file)
        parser = CkyParser(grammar)

        toks = ['flights', 'from', 'miami', 'to', 'cleveland', '.']
        print(parser.is_in_language(toks))

        # toks = ['miami', 'flights', 'cleveland', 'from', 'to',  '.']
        # print(parser.is_in_language(toks))

        table, probs = parser.parse_with_backpointers(toks)

        ## print table
        # for i in range(0, len(toks) + 1):
        #
        #     for j in range(i + 1, len(toks) + 1):
        #         print(i, j, " ", table[(i, j)], end="  ####  ")
        #     print()

        assert check_table_format(table)
        assert check_probs_format(probs)

        print(get_tree(table, 0, 6, grammar.startsymbol))

        # test_dict = defaultdict(float)
        # print(test_dict["A"]) # 0.0
