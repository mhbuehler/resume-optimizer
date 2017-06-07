# Adapted from: https://github.com/sujitpal/mlia-examples/blob/master/src/salary_pred/rake_nltk.py
# Which was adapted from: github.com/aneesha/RAKE/rake.py
from __future__ import division
import sys
import operator
import string
import nltk
from stemming.porter2 import stem

def is_punctuation(word):
    return len(word) == 1 and word in string.punctuation

def is_numeric(word):
    try:
        float(word) if '.' in word else int(word)
        return True
    except ValueError:
        return False


class KeywordExtractor:
    """Extracts keywords and keyphrases from text input"""
    def __init__(self):
        self.stopwords = set(nltk.corpus.stopwords.words())
        self.top_fraction = 1

    def _generate_candidate_keywords(self, sentences, max_length=3):
        """Creates a list of candidate keywords, or phrases of at most max_length words, from a set of sentences"""
        phrase_list = []
        for sentence in sentences:
            words = map(lambda x: "|" if x in self.stopwords else x,
                        nltk.word_tokenize(sentence.lower()))
            phrase = []
            for word in words:
                if word == "|" or is_punctuation(word):
                    if len(phrase) > 0:
                        if len(phrase) <= max_length:
                            phrase_list.append(phrase)
                        phrase = []
                else:
                    phrase.append(word)

        return phrase_list

    def _calculate_word_scores(self, phrase_list):
        """Scores words according to frequency and tendency to appear in multi-word key phrases"""
        word_freq = nltk.FreqDist()
        word_multiplier = nltk.FreqDist()
        for phrase in phrase_list:
            # Give a higher score if word appears in multi-word candidates
            multi_word = min(2, len(filter(lambda x: not is_numeric(x), phrase)))
            for word in phrase:
                # Normalize by taking the stem
                word_freq[stem(word)] += 1
                word_multiplier[stem(word)] += multi_word
        for word in word_freq.keys():
            word_multiplier[word] = word_multiplier[word] / float(word_freq[word])  # Take average
        word_scores = {}
        for word in word_freq.keys():
            word_scores[word] = word_freq[word] * word_multiplier[word]

        return word_scores

    def _calculate_phrase_scores(self, phrase_list, word_scores, metric='avg'):
        """Scores phrases by taking the average, sum, or max of the scores of its words"""
        phrase_scores = {}
        for phrase in phrase_list:
            phrase_score = 0
            if metric in ['avg', 'sum']:
                for word in phrase:
                    phrase_score += word_scores[stem(word)]
                phrase_scores[" ".join(phrase)] = phrase_score
                if metric == 'avg':
                    phrase_scores[" ".join(phrase)] = phrase_score / float(len(phrase))
            elif metric == 'max':
                for word in phrase:
                    phrase_score = word_scores[stem(word)] if word_scores[stem(word)] > phrase_score else phrase_score
                phrase_scores[" ".join(phrase)] = phrase_score

        return phrase_scores

    def extract(self, text, max_length=3, metric='avg', incl_scores=False):
        """Extract keywords and keyphrases from input text in descending order of score"""
        sentences = nltk.sent_tokenize(text)
        phrase_list = self._generate_candidate_keywords(sentences, max_length=max_length)
        word_scores = self._calculate_word_scores(phrase_list)
        phrase_scores = self._calculate_phrase_scores(phrase_list, word_scores, metric=metric)
        sorted_phrase_scores = sorted(phrase_scores.iteritems(), key=operator.itemgetter(1), reverse=True)
        n_phrases = len(sorted_phrase_scores)

        if incl_scores:
            return sorted_phrase_scores[0:int(n_phrases/self.top_fraction)]
        else:
            return map(lambda x: x[0], sorted_phrase_scores[0:int(n_phrases/self.top_fraction)])

def main(file_path):
    """Extracts keywords in order of relevance from a text file. Requires one command line argument.

    Args:
        file_path (str): path to text file
    """
    kwe = KeywordExtractor()
    with open(file_path) as file:
        content = unicode(file.read(), errors='ignore')
    keywords = kwe.extract(content, incl_scores=True)
    print keywords

if __name__ == "__main__":
    main(*sys.argv[1:])