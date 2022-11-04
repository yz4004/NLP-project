import sys
from collections import defaultdict


class DependencyEdge(object):
    """
    Represent a single dependency edge: 
    """

    def __init__(self, ident, word, pos, head, deprel):
        self.id = ident
        self.word = word
        self.pos = pos
        self.head = head
        self.deprel = deprel

    def print_conll(self):
        return "{d.id}\t{d.word}\t_\t_\t{d.pos}\t_\t{d.head}\t{d.deprel}\t_\t_".format(d=self)


def parse_conll_relation(s):
    fields = s.split('\t')
    ident_s, word, lemma, upos, pos, feats, head_s, deprel, deps, misc = fields  # ident_s, word, pos, head_s, deprel
    ident = int(ident_s)
    head = int(head_s)
    return DependencyEdge(ident, word, pos, head, deprel)


class DependencyStructure(object):

    def __init__(self):
        self.deprels = {}
        # {deprel.id : deprel, ... }
        # deprel is DependencyEdge (5 elements: id/ident, word, pos, head, deprel)
        self.root = None
        self.parent_to_children = defaultdict(list)
        # {head_id : child_id_1, head_id : child_id_2},  all the
        # children of head

    def add_deprel(self, deprel):
        self.deprels[deprel.id] = deprel
        self.parent_to_children[deprel.head].append(deprel.id)
        if deprel.head == 0:
            self.root = deprel.id

    def __str__(self):
        for k, v in self.deprels.items():
            print(v.print_conll())
            # yield v.print_conll()
        return "FINISH"

    def print_tree(self, parent=None):
        if not parent:
            return self.print_tree(parent=self.root)

        if self.deprels[parent].head == parent:
            return self.deprels[parent].word

        children = [self.print_tree(child) for child in self.parent_to_children[parent]]
        child_str = " ".join(children)
        return ("({} {})".format(self.deprels[parent].word, child_str))

    def words(self):
        return [None] + [x.word for (i, x) in self.deprels.items()]  # DependencyEdge() x ~ deprel

    def pos(self):
        return [None] + [x.pos for (i, x) in self.deprels.items()]

    def print_conll(self):
        deprels = [v for (k, v) in sorted(self.deprels.items())]
        return "\n".join(deprel.print_conll() for deprel in deprels)


def conll_reader(input_file):
    current_deps = DependencyStructure()
    while True:
        line = input_file.readline().strip()  # a string with no space at the beginning & end
        if not line and current_deps: # ??
            yield current_deps
            current_deps = DependencyStructure()
            line = input_file.readline().strip()
            # strip 只剥去收尾的空格 tab 换行（可以括号里另指定）， 这时 line 是一个长string，由9个tab隔开的10个词的成分
            # 其中 只有 5 个是我们需要的，句子 string 喂给 parse_conll_relation 会提出这需要的 5 个成分 装入一个 DependencyEdge 对象中
            # 注意 这个edge 只是一个词的 parse 结构，它应作为 这个句子的 parse 结构的一部分。
            # 注意 在这之前就创建了 一个 DependencyStructur() 对象 current_deps 这个 edge 对象 由 add_deprels() 加入到 DependencyStructure
            # 只有遇到空行 才是句子的结束，然后开启下一行的第一个词

            if not line:
                break
        current_deps.add_deprel(parse_conll_relation(line))  ## DependencyEdge add -> DependencyStructure


if __name__ == "__main__":
    with open(sys.argv[1], 'r') as in_file:
        relations = set()
        for deps in conll_reader(in_file):
            for deprel in deps.deprels.values():
                relations.add(deprel.deprel)
            print(deps.print_tree())
