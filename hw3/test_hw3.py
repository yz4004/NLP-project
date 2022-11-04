import sys
from conll_reader import DependencyStructure, parse_conll_relation

if __name__ == "__main__":

    print("hello")
    ## take a look at words.vocab, pos.vocab
    # with open('data/words.vocab') as word_file: #, open('pos.vocab','r') as pos_file:
    #
    #     for index, line in enumerate(word_file):
    #         print(index, line)

    # with open('data/pos.vocab','r') as pos_file:
    #
    #     for index, line in enumerate(pos_file):
    #         print(index, line)

    ## 打印行的方法：
    # https://blog.csdn.net/qdPython/article/details/106160272?msclkid=74b26c58ad8c11ecbee59ef9e1d0a91e

    # s = " ab c\n"
    #
    # print(s.split())
    # print(s.strip().split())

    # a = ["aa", "dd"]
    # print( dict((index, x) for (index, x) in enumerate(a))  )
    # print(list((index, x) for (index, x) in enumerate(a)))
    # print([(index, x) for (index, x) in enumerate(a)])
    # # print((index, x) for (index, x) in enumerate(a))
    # # <generator object <genexpr> at 0x000001CE2488D350> generator 放入 dict 可迭代是一方面，每个迭代输出必须是 二元tuple
    #
    # # print(dict(("1","b" )))
    # print(dict([("1", "b")] ) )  # dict() 里接受  [(,)] 字典套二元tuple
    # # print(dict([("1", "b", "c") ]))
    # # ValueError: dictionary update sequence element #0 has length 3; 2 is required 要求二元tuple
    #
    # # 更多用法
    # # https: // www.runoob.com / python / python - func - dict.html?msclkid = 32
    # # bd3c55adb711eca0f55c4a4c857ac5

    ##
    with open('data/dev.conll') as dev_conll:

        line = dev_conll.readline().strip()

        current_deps = DependencyStructure()
        if current_deps: print("yes")

        current_deps.add_deprel(parse_conll_relation(line))

        # current_deps.__str__()
        print(current_deps)

        if current_deps:
            print("yes")

        # print( not False or True)  # true
        # print(not (False or True))  # false
