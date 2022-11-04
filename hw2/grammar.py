"""
COMS W4705 - Natural Language Processing
Homework 2 - Parsing with Context Free Grammars 
Yassine Benajiba
"""
import math
import sys
from collections import defaultdict
from math import fsum

# from cky import CkyParser


class Pcfg(object):
    """
    Represent a probabilistic context free grammar. 
    """

    def __init__(self, grammar_file):
        self.rhs_to_rules = defaultdict(list)
        self.lhs_to_rules = defaultdict(list)
        self.startsymbol = None
        self.read_rules(grammar_file)

        if self.verify_grammar():
            print("Safe -- right format")
        else:
            print("Not safe -- wrong format")

    def read_rules(self, grammar_file):

        for line in grammar_file:
            line = line.strip()
            if line and not line.startswith("#"):
                if "->" in line:
                    rule = self.parse_rule(line.strip())
                    lhs, rhs, prob = rule
                    self.rhs_to_rules[rhs].append(rule)
                    self.lhs_to_rules[lhs].append(rule)
                else:
                    startsymbol, prob = line.rsplit(";")
                    self.startsymbol = startsymbol.strip()

    def parse_rule(self, rule_s):
        lhs, other = rule_s.split("->")
        lhs = lhs.strip()
        rhs_s, prob_s = other.rsplit(";", 1)
        prob = float(prob_s)
        rhs = tuple(rhs_s.strip().split())
        return (lhs, rhs, prob)

    def verify_grammar(self):
        """
        Return True if the grammar is a valid PCFG in CNF.
        Otherwise return False. 
        """
        # TODO, Part 1

        result = True
        for key, value in self.lhs_to_rules.items():
            if not key.isupper():
                result = False
                break
            for tuple_grammar in value:
                if len(tuple_grammar[1]) > 1:
                    for i in tuple_grammar[1]:
                        if not i.isupper():
                            result = False
                            print(i)
                            break
                if not result:
                    break

            temp_list = list(map(list, zip(*value)))
            # print(temp_list)
            temp_sum = math.fsum(temp_list[-1])
            # print(temp_sum)
            if not math.isclose(temp_sum, 1, abs_tol=1e-5):
                result = False
                break

        return result


if __name__ == "__main__":
    with open(sys.argv[1], 'r') as grammar_file:
        grammar = Pcfg(grammar_file)

    ### test code
    # with open("atis3.pcfg", 'r') as grammar_file:
    #     grammar = Pcfg(grammar_file)     # grammar = Pcfg(r"C:\Users\31557\a_projects 2\4705\hw2\atis3.pcfg")

    # print(grammar.startsymbol)

    # for key,value in grammar.rhs_to_rules.items():
    #     print(key, value)

    # print(grammar.rhs_to_rules.get("BSDFdfdf"))  # None
    # print(grammar.rhs_to_rules.get(('TO', 'CLEVELAND')))  # None
    # print(grammar.rhs_to_rules.get(('FROM', 'NP')))  # None
    # print(grammar.rhs_to_rules.get(('cleveland',)))  # None
    # print(grammar.rhs_to_rules.get(('miami',)))  # None
    # print(grammar.rhs_to_rules.get(('TO', 'NP')))  # None
    # print(grammar.rhs_to_rules.get(('CLEVELAND',)))  # None


