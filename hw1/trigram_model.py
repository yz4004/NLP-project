import sys
from collections import defaultdict
import math
import random
import os
import os.path
import numpy as np

"""
COMS W4705 - Natural Language Processing
Homework 1 - Programming Component: Trigram Language Models
Yassine Benajiba
"""


def corpus_reader(corpusfile, lexicon=None):
    with open(corpusfile, 'r') as corpus:
        for line in corpus:
            if line.strip():  # strip去空格， 目的是判断空行
                sequence = line.lower().strip().split()
                if lexicon:
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else:
                    yield sequence


def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence:
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)


def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """
    result = []

    temp_sequence = sequence.copy()

    # if n == 1:
    #     temp_sequence.insert(0, "START")
    # else:
    #     for i in range(n - 1):
    #         temp_sequence.insert(i, "START")
    for i in range(n):
        temp_sequence.insert(i, "START")
    temp_sequence.append("STOP")

    for i in range(len(temp_sequence) - n + 1):  # should > len(seq)

        result.append(tuple(temp_sequence[i:i + n]))

    return result


class TrigramModel(object):

    def __init__(self, corpusfile):

        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")

        self.total_number = None # update in raw_unigram_probability

        self.corpusfile = corpusfile

        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)

        self.total_number = 0
        for unigram in self.unigramcounts:
            self.total_number += self.unigramcounts[unigram]

        # print(self.total_number, self.unigramcounts.get(("START",)) ,self.unigramcounts.get(("STOP",)))

        self.total_number = self.total_number - self.unigramcounts.get(("START",)) - self.unigramcounts.get(("STOP",))

        # print(self.total_number)


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """

        self.unigramcounts = defaultdict(int)  # might want to use defaultdict or Counter instead
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)

        ##Your code here

        for sentence in corpus:
            for unigram in get_ngrams(sentence, 1):
                self.unigramcounts[unigram] += 1

            for bigram in get_ngrams(sentence, 2):
                self.bigramcounts[bigram] += 1

            for trigram in get_ngrams(sentence, 3):
                self.trigramcounts[trigram] += 1

        # print("HERE", self.bigramcounts.get(('my', 'school')))
        # print( "HERE", self.bigramcounts.get( ('school', 'you') ))
        # print("HERE", self.trigramcounts.get(('school', 'you', 'often')))

        # modifying sequence in count_ngram needs to copy! or it will corrupt sequence for later use
        # print("mark1",self.unigramcounts.get(('START',)))
        # print("mark2",self.bigramcounts.get(('START', 'START')))
        # print("mark3",self.trigramcounts.get(('START', "START", "START")))
        #

        # del self.unigramcounts[("START",)]
        # print("mark1", self.unigramcounts.get(('START',)))

        return

    def raw_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """

        bigram = trigram[0:2]

        # print(trigram, bigram, trigram[0:1])
        # print("count:", self.bigramcounts[bigram])

        return self.trigramcounts[trigram] / self.bigramcounts[bigram]  # ??

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        unigram = (bigram[0],)  # bigram[0:1]

        return self.bigramcounts[bigram] / self.unigramcounts[unigram]

    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        # if self.total_number == None:
        #     total_number = 0
        #     for unigram in self.unigramcounts:
        #         total_number += self.unigramcounts[unigram]
        #     self.total_number = total_number

        # hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.  
        return self.unigramcounts[unigram] / self.total_number

    def generate_sentence(self, t=20):
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """

        result = []

        prob1 = [[index, self.raw_trigram_probability(trigram)] for index, trigram in
                 enumerate(self.trigramcounts)
                 if trigram[0:2] == ('START', 'START')]
        # print(prob1, " ", np.sum(np.array(prob1)[:,1] ) )
        draw = np.random.multinomial(1, np.array(prob1)[:, 1], 1).flatten()

        current_index = list(self.trigramcounts.values())[np.where(draw == 1)[0][0]]

        # print(current_index)
        current_word = list(self.trigramcounts.keys())[current_index][2]
        # print(current_word)
        bigram = ('START', current_word)

        current_len = 1

        result.append(current_word)

        while (current_len < 20) and (current_word != 'STOP'):
            print(bigram)
            prob = [[index, self.raw_trigram_probability(trigram)] for index, trigram in
                    enumerate(self.trigramcounts)
                    if trigram[0:2] == bigram]
            draw = np.random.multinomial(1, np.array(prob1)[:, 1], 1).flatten()
            current_index = list(self.trigramcounts.values())[np.where(draw == 1)[0][0]]
            current_word = list(self.trigramcounts.keys())[current_index][2]
            bigram = (bigram[1], current_word)
            result.append(current_word)

            current_len += 1

        return result

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1 / 3.0
        lambda2 = 1 / 3.0
        lambda3 = 1 / 3.0
        # print(self.raw_unigram_probability(trigram[0:1]), self.raw_bigram_probability(trigram[0:2]))

        prob = 0

        if self.unigramcounts.get(trigram[0:1]) != None:
            prob += lambda1 * self.raw_unigram_probability(trigram[0:1])
        if self.bigramcounts.get(trigram[0:2]) != None:
            prob += lambda2 * self.raw_bigram_probability(trigram[0:2])
        if self.trigramcounts.get(trigram) != None:
            prob += lambda3 * self.raw_trigram_probability(trigram)

        # print(self.raw_unigram_probability(trigram[0:1]),
        #       self.raw_bigram_probability(trigram[0:2]),
        #       self.raw_trigram_probability(trigram))


        # prob = lambda1 * self.raw_unigram_probability(trigram[0:1])
        # + lambda2 * self.raw_bigram_probability(trigram[0:2])
        # + lambda3 * self.raw_trigram_probability(trigram)

        return prob

    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        trigrams = get_ngrams(sentence, 3)  # sentence is a list of words, trigrams is a list of trigrams

        log_prob = 0
        # print(trigrams)
        # print(math.log2(self.smoothed_trigram_probability(trigrams[0])))

        for trigram in trigrams[1:]:  # skip the first element of trigrams:("START","START","START")
            # print(trigram)
            log_prob += math.log2(self.smoothed_trigram_probability(trigram))
            # print(trigram, log_prob) # pow(2,log_prob))

        return log_prob
        # return float("-inf")

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """

        l = 0
        M = 0
        # generator = corpus_reader(self.corpusfile, self.lexicon)
        # for sentence in generator:
        #     l += self.sentence_logprob(sentence)

        for sentence in corpus:
            M += len(sentence) + 2
            l += self.sentence_logprob(sentence)

        pp = np.exp2(-l/M)
        return pp


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):
    model1 = TrigramModel(training_file1)
    model2 = TrigramModel(training_file2)
    total = 0
    correct = 0

    for f in os.listdir(testdir1):
        pp = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))  # train high ~ test high
        # ..
        pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))  # train low ~ test high
        total += 1
        # print(f)
        # rint("high",pp, pp2)
        if pp < pp2:
            correct += 1

    # high_total = total
    # high_correct = correct


    for f in os.listdir(testdir2):
        pp = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))  # train low ~ test low
        # ..
        pp1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))  # train high ~ test low
        total += 1
        print("low", pp, pp1)
        if pp < pp1:
            correct += 1
    # print("HIGH TOTAL & CORRECT", high_total, high_correct)
    # print("LOW TOTAL & CORRECT", total - high_total, correct - high_correct)
    # print(total, correct)

    return correct / total


if __name__ == "__main__":
    # model = TrigramModel(sys.argv[1])
    model = TrigramModel(r".\hw1_data\brown_train.txt")

    print(model.trigramcounts[('START', 'START', 'the')])
    print(model.bigramcounts[('START', 'the')])
    print(model.unigramcounts[('the',)])

    print(model.unigramcounts[("UNK",)])

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt.

    # my test
    # corpus_reader 读取的 每行是是什么样子
    # generator = corpus_reader(r".\hw1_data\brown_train.txt")
    # for sequence in generator:
    #     print(sequence)

    # Testing perplexity:
    # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    dev_corpus = corpus_reader(r".\hw1_data\brown_test.txt", model.lexicon)
    pp = model.perplexity(dev_corpus)
    print(pp)

    train_corpus = corpus_reader(r".\hw1_data\brown_train.txt", model.lexicon)
    print(model.perplexity(train_corpus))

    # Essay scoring experiment:
    # acc = essay_scoring_experiment('train_high.txt', 'train_low.txt", "test_high", "test_low")




    # test get_ngram()
    # g = corpus_reader(r".\hw1_data\brown_train.txt", model.lexicon)
    # for i in range(5):
    #     s = next(g)
    #     print(s, "\n", get_ngrams(s, 1))
    # print("######################")
    #
    # for i in range(5):
    #     s = next(g)
    #     print(s, "\n", get_ngrams(s, 2))
    # print("######################")
    #
    # for i in range(5):
    #     s = next(g)
    #     print(s, "\n", get_ngrams(s, 3))
    # print("######################")



    acc = essay_scoring_experiment('./hw1_data/ets_toefl_data/train_high.txt',
                                   './hw1_data/ets_toefl_data/train_low.txt',
                                   './hw1_data/ets_toefl_data/test_high',
                                   './hw1_data/ets_toefl_data/test_low')
    print(acc)
