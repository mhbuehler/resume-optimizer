import sys
from collections import Counter
from itertools import product
import string
import enchant
from textblob import TextBlob, Word
from textblob.wordnet import VERB
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
import keyword_extractor

def read_file(file_path):
    with open(file_path) as file:
        content = unicode(file.read(), errors='ignore')

    return content.decode('utf8').encode('ascii', 'ignore')

def read_resume(file_path):
    resume_data = {}
    resume_data['content'] = TextBlob(read_file(file_path))
    resume_data['skills'] = extract_skills(str(resume_data['content']))
    resume_data['actions'] = extract_actions(resume_data['content'])

    return resume_data

def read_job(file_path):
    """Reads a text file with the job title on the first line and description following"""
    job_data = {}
    job_data['description'] = TextBlob(read_file(file_path))
    job_data['title'] = str(job_data['description']).split('\n')[0]
    job_data['skills'] = extract_skills(str(job_data['description']))
    job_data['noun_phrases'] = extract_nouns(job_data['description'])
    job_data['actions'] = extract_actions(job_data['description'])
    job_data['acronyms'] = extract_acronyms(job_data['description'])
    job_data['value_sentences'] = extract_value_sentences(job_data['description'])

    return job_data

def get_all_skills():
    skills = []
    with open('data/all_linkedin_skills.txt') as input_file:
        for line in input_file:
            skills.append(line.strip().lower())

    return skills

def extract_skills(text_string):
    all_skills = get_all_skills()
    extractor = keyword_extractor.KeywordExtractor()
    skills = [s for s in extractor.extract(str(text_string), incl_scores=True) if s[0] in all_skills]

    return skills

def extract_nouns(textblob):
    """Creates a list of noun phrases in descending order of frequency"""
    phrases = textblob.noun_phrases
    counts = []
    for phrase in phrases:
        if phrase.lower() not in [p[0] for p in counts]:
            counts.append((phrase, textblob.noun_phrases.count(phrase)))

    return sorted(counts, key=lambda x: x[1], reverse=True)

def extract_actions(textblob):
    """Creates a list of verbs (action words) in descending order of frequency"""
    verbs = [t for t in textblob.tags if t[1][:2]=='VB']
    verb_roots = [v[0].lemmatize('v').lower() for v in verbs]
    counter = Counter(verb_roots)

    return sorted(counter.items(), key=lambda x: x[1], reverse=True)
    
def extract_acronyms(textblob):
    """Creates a list of words beginning with at least 2 capital letters that are not regular English words,
    in descending order of frequency. enchant dictionary returns True if word is an English word."""
    d = enchant.Dict("en_US")
    words = textblob.words
    counts = []
    for word in words:
        if len(word) > 1:
            if word[0].isupper() and word[1].isupper() and word not in [p[0] for p in counts]:
                if not d.check(word):
                    counts.append((word, textblob.words.count(word)))

    return counts

def extract_value_sentences(textblob, polarity_threshold=0.3, subjectivity_threshold=0.5):
    """Creates a list of sentences with high polarity/subjectivity (i.e. "value" sentences)"""
    value_sentences = []
    for s in textblob.sentences:
        if s.sentiment.polarity > polarity_threshold and s.sentiment.subjectivity > subjectivity_threshold:
            value_sentences.append((s, s.sentiment.polarity + s.sentiment.subjectivity))

    return sorted(value_sentences, key=lambda x: x[1], reverse=True)

def main(job_file_path):
    """Imports a job description and extracts skills, acronyms, noun phrases, action words,
    and sentences with high sentiment.

    Args:
        job_file_path (str): path to job description text file
    """
    job = read_job(job_file_path)
    print "Job Title: {}\n\n".format(job['title'])
    print "Skills: {}\n\n".format(job['skills'])
    print "Acronyms: {}\n\n".format(job['acronyms'])
    print "Noun Phrases: {}\n\n".format(job['noun_phrases'])
    print "Verbs: {}\n\n".format(job['actions'])
    print "Value Sentences: {}\n\n".format(job['value_sentences'])

if __name__ == "__main__":
    main(*sys.argv[1:])

