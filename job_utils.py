import os
import sys
from collections import Counter
from textblob import TextBlob, Word
from textblob.wordnet import VERB
from sklearn.feature_extraction.text import TfidfVectorizer


def read_resume(file_path):
    with open(file_path) as file:
        content = unicode(file.read(), errors='ignore')

    return TextBlob(content.decode('utf8').encode('ascii', 'ignore'))

def read_job_description(file_path):
    """Reads a text file with the job title on the first line and description following
    
    Args:
        file_path (str): path to text file (.txt format) containing job title and description
        
    Returns:
        dictionary containing job title and job description as a TextBlob
    """
    job_data = {}
    with open(file_path) as file:
        job_data['title'] = unicode(file.readline(), errors='ignore').replace('\n', '')
        description = unicode(file.read(), errors='ignore')
        job_data['description'] = TextBlob(description.decode('utf8').encode('ascii','ignore'))
    job_data['keywords'] = extract_keywords(job_data['description'])
    job_data['value_sentences'] = extract_value_sentences(job_data['description'])
    job_data['actions'] = extract_actions(job_data['description'])
    return job_data

def extract_keywords(textblob):
    """Creates a list of noun phrases (or keywords) in descending order of frequency"""
    phrases = textblob.noun_phrases
    counts = []
    for phrase in phrases:
        if phrase.lower() not in [p[0] for p in counts]:
            counts.append((phrase, textblob.noun_phrases.count(phrase)))

    return sorted(counts, key=lambda x: x[1], reverse=True)

def extract_value_sentences(textblob, polarity_threshold=0.3, subjectivity_threshold=0.5):
    """Creates a list of sentences with high polarity/subjectivity (i.e. "value" sentences)"""
    value_sentences = []
    for s in textblob.sentences:
        if s.sentiment.polarity > polarity_threshold and s.sentiment.subjectivity > subjectivity_threshold:
            value_sentences.append((s, s.sentiment.polarity + s.sentiment.subjectivity))

    return sorted(value_sentences, key=lambda x: x[1], reverse=True)

def extract_actions(textblob):
    """Creates a list of verbs (action words) in descending order of frequency"""
    verbs = [t for t in textblob.tags if t[1][:2]=='VB']
    verb_roots = [v[0].lemmatize('v').lower() for v in verbs]
    counter = Counter(verb_roots)

    return sorted(counter.items(), key=lambda x: x[1], reverse=True)

def compute_similarity(textblob_1, textblob_2):
    documents = [str(textblob_1), str(textblob_2)]
    vector = TfidfVectorizer(min_df=1)
    similarity_vector = vector.fit_transform(documents)

    return (similarity_vector * similarity_vector.T).A[0, 1]

def suggest_synonyms(words, target_words):
    suggestions = []
    word_synonyms = [(Word(w[0]).get_synsets(pos=VERB), w[1]) for w in target_words]
    for w in words:
        found = False
        synset = (Word(w[0]).get_synsets(pos=VERB), w[1])
        if len(synset[0]):
            for synonym in [s for s in word_synonyms if len(s[0])]:
                similarity = synset[0][0].path_similarity(synonym[0][0])
                if similarity == 1.0:
                    found = True
                if 1.0 > similarity > 0.4 and not found:
                    suggestions.append((synset[0][0].name().split(".")[0], synonym[0][0].name().split(".")[0]))

    return suggestions

def main(job_file_path, resume_path=None):
    job = read_job_description(job_file_path)
    print "Job Title: {title}\n\nKeywords: {keywords}\n\nValue Sentences: {value_sentences}\n\nActions: {actions}".format(**job)

    if resume_path:
        resume = read_resume(resume_path)
        print "\nResume Similarity Index: {}".format(compute_similarity(resume, job['description']))
        suggestions = suggest_synonyms(extract_actions(resume), job['actions'])
        print ''
        for suggestion in suggestions:
            print 'Where you use', suggestion[0].upper(), 'consider using', suggestion[1].upper()


if __name__ == "__main__":
    main(*sys.argv[1:])

