import os
import sys
from collections import Counter
from textblob import TextBlob


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
    
    return job_data

def extract_keywords(job):
    """Creates a list of noun phrases (or keywords) in descending order of frequency"""
    job_data = job.copy()
    phrases = job_data['description'].noun_phrases
    counts = []
    for phrase in phrases:
        if phrase.lower() not in [p[0] for p in counts]:
            counts.append((phrase, job_data['description'].noun_phrases.count(phrase)))
    job_data['keywords'] = sorted(counts, key=lambda x: x[1], reverse=True)
    
    return job_data

def extract_value_sentences(job, polarity_threshold=0.3, subjectivity_threshold=0.5):
    """Creates a list of sentences with high polarity/subjectivity (i.e. "value" sentences)"""
    job_data = job.copy()
    value_sentences = []
    for s in job_data['description'].sentences:
        if s.sentiment.polarity > polarity_threshold and s.sentiment.subjectivity > subjectivity_threshold:
            value_sentences.append((s, s.sentiment.polarity + s.sentiment.subjectivity))
    job_data['value_sentences'] = sorted(value_sentences, key=lambda x: x[1], reverse=True)
    
    return job_data

def extract_actions(job):
    """Creates a list of verbs (action words) in descending order of frequency"""
    job_data = job.copy()
    verbs = [t for t in job_data['description'].tags if t[1][:2]=='VB']
    verb_roots = [v[0].lemmatize('v').lower() for v in verbs]
    counter = Counter(verb_roots)
    job_data['actions'] = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    
    return job_data

def main(file_path):
    job = read_job_description(file_path)
    job = extract_keywords(job)
    job = extract_value_sentences(job)
    job = extract_actions(job)
    print "Job Title: {title}\n\nKeywords: {keywords}\n\nValue Sentences: {value_sentences}\n\nActions: {actions}".format(**job)

if __name__ == "__main__":
    main(sys.argv[1])

