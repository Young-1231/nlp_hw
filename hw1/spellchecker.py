from collections import Counter
import numpy as np
from nltk.corpus import reuters
import re
import chardet

class NgramModel:
    """
    This class builds n-gram language models from a given corpus.
    """

    def __init__(self, corpus, n):
        """
        Initializes the n-gram model with the provided corpus and n-gram order.
        """
        self.term_count = dict()
        self.ngram_count = dict()
        self.build(corpus, n)

    def build(self, corpus, n):
        """
        Builds the n-gram language model from the corpus.
        """
        term_counter = Counter()
        ngram_counter = Counter()

        for sentence in corpus:
            sentence = ["<s>"] + sentence  # Add sentence start marker
            for i in range(len(sentence) - n + 1):
                term = sentence[i]
                ngram = " ".join(sentence[i:i + n])

                term_counter.update([term])
                ngram_counter.update([ngram])

        self.term_count = dict(term_counter)
        self.ngram_count = dict(ngram_counter)

    def calculate_smoothed_probability(self, bigram, term, V):
        """
        Calculates the smoothed log probability of a bigram using Laplace smoothing.
        """
        bigram_freq = self.ngram_count.get(bigram, 0) + 1
        term_freq = self.term_count.get(term, 0) + V
        return np.log(bigram_freq / term_freq)


class CandidatesGenerator:
    """
    Implements spelling correction using bigram language models and error channel probabilities.
    """

    def __init__(self, vocabulary):
        self.vocabulary = set(vocabulary)
        self.alphabet = "abcdefghijklmnopqrstuvwxyz"

    def generate_candidates(self, word, max_distance=1):
        """
        Generates candidate corrections for a misspelled word within a specified edit distance.
        """
        candidates = {word}
        for _ in range(max_distance):
            new_candidates = set()
            for candidate in candidates:
                new_candidates.update(self.generate_edits(candidate))
            candidates.update(new_candidates)
        return list(candidates - {word})

    def generate_edits(self, word):
        """
        Generates all possible edits (insertions, deletions, substitutions, transpositions)
        for a given word within a single edit distance.
        """
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        edits = (
                [L + c + R for L, R in splits for c in self.alphabet] +
                [L + R[1:] for L, R in splits if R] +
                [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1] +
                [L + c + R[1:] for L, R in splits if R for c in self.alphabet] +
                [word[:i] + word[j] + word[i + 1:j] + word[i] + word[j + 1:] for i in range(len(word)) for j in
                 range(i + 1, len(word))]
        )
        return [edit for edit in set(edits) if edit in self.vocabulary]


class ChannelProbability:
    """
    This class parses and handles error channel probabilities.
    """

    def __init__(self, filename):
        self.channel_prob = self.parse_channel_probabilities(filename)

    def parse_channel_probabilities(self, filename):
        """
        Parses the error channel probabilities from a file.
        """
        channel_prob = Counter()
        total_errors = 0
        with open(filename, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']

        with open(filename, 'r', encoding=encoding) as file:
            for line in file:
                count = int(re.findall(r'\d+', line)[-1])
                line = re.sub(r'\d+', '', line).replace('\t', '').strip()
                correct, mistake = line.split('|')

                channel_prob[(correct, mistake)] += count
                total_errors += count

        channel_prob = {k: v / total_errors for k, v in channel_prob.items()}
        return channel_prob


class SpellChecker:
    """
    This class processes lines of text, applying spelling correction with word frequency optimization.
    """

    def __init__(self, vocab_file, corpus, channel_prob_file, max_distance=1, ngram=2):
        self.vocabulary = {line.rstrip() for line in open(vocab_file)}
        self.ngram_model = NgramModel(corpus, ngram)
        self.channel_prob = ChannelProbability(channel_prob_file).channel_prob
        self.corrector = CandidatesGenerator(self.vocabulary)
        self.max_distance = max_distance
        self.V = len(self.ngram_model.term_count)

    def process_line(self, line):
        """
        Processes a single line of text, applying spelling correction with word frequency optimization.
        """
        line = re.sub(r"([,])([^\d])", r" \1 \2", line)
        line = re.sub(r"([^s])(['])", r"\1 \2", line)
        line = re.sub(r"([s])(['])", r"\1 \2 ", line)
        line = re.sub(r"([.]$)", r" \1 ", line)

        items = line.split("\t")
        sentence = items[2].split() if len(items) > 2 else items[0].split()
        corrected_sentence = sentence.copy()

        for j, word in enumerate(sentence):
            if word not in self.vocabulary:
                candidates = self.corrector.generate_candidates(word, self.max_distance)
                if not candidates:
                    continue

                best_candidate = word
                max_prob = float('-inf')

                for candidate in candidates:
                    prob = np.log(self.channel_prob.get((candidate, word), 0.0001))

                    if j > 0:
                        forward_bigram = f"{sentence[j - 1]} {candidate}"
                        prob += self.ngram_model.calculate_smoothed_probability(forward_bigram, sentence[j - 1], self.V)

                    if j + 1 < len(sentence):
                        backward_bigram = f"{candidate} {sentence[j + 1]}"
                        prob += self.ngram_model.calculate_smoothed_probability(backward_bigram, candidate, self.V)

                    if prob > max_prob:
                        max_prob = prob
                        best_candidate = candidate

                if len(word) > 1:
                    corrected_sentence[j] = best_candidate

        corrected_sentence = " ".join(corrected_sentence)
        corrected_sentence = re.sub(r"\s*(['])\s*", r"\1", corrected_sentence)
        corrected_sentence = re.sub(r"(s')", r"\1 ", corrected_sentence)
        corrected_sentence = re.sub(r"\s([.])\s", r"\1", corrected_sentence)
        corrected_sentence = re.sub(r"\s([,])", r"\1", corrected_sentence)
        corrected_sentence = re.sub(r"(\d)([,])\s+(\d)", r"\1\2\3", corrected_sentence)

        if corrected_sentence[-1] != ".":
            corrected_sentence += "."

        return corrected_sentence


def main():
    corpus = reuters.sents(categories=reuters.categories())
    spellchecker = SpellChecker('vocab.txt', corpus, 'count_1edit.txt', max_distance=1, ngram=2)
    file_path = "testdata.txt"

    with open(file_path, "r") as file, open("result.txt", "w") as output_file:
        for line_num, line in enumerate(file, start=1):
            corrected_sentence = spellchecker.process_line(line)
            output_file.write(f"{line_num}\t{corrected_sentence}\n")


if __name__ == "__main__":
    main()
