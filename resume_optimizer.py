import nltk
from textblob import TextBlob, Word
from textblob.wordnet import VERB
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
import keyword_extractor
from utils import *


class ResumeOptimizer:
    """"""
    def __init__(self, job_path, resume_path):
        self.job = read_job(job_path)
        self.resume = read_resume(resume_path)

    @property
    def similarity(self):
        return self._compute_similarity(self.job['description'], self.resume['content'])

    def _compute_similarity(self, textblob_1, textblob_2):
        documents = [str(textblob_1), str(textblob_2)]
        vector = TfidfVectorizer(min_df=1)
        similarity_vector = vector.fit_transform(documents)

        return (similarity_vector * similarity_vector.T).A[0, 1]

    def _find_similar_skills(self, target_skill, skills, threshold=0.5):
        list1 = target_skill.split(' ')
        syns1 = set(ss for word in list1 for ss in wordnet.synsets(word))
        scores = {}
        skill_list = []
        for skill in skills:
            scores[skill] = (0, None, None)
            list2 = skill.split(' ')
            syns2 = set(ss for word in list2 for ss in wordnet.synsets(word))
            for s1, s2 in product(syns1, syns2):
                if wordnet.wup_similarity(s1, s2) > scores[skill][0]:
                    scores[skill] = (wordnet.wup_similarity(s1, s2), s1, s2)

            if scores[skill][0] >= threshold:
                skill_list.append((skill, scores[skill][0]))

        return sorted(skill_list, key=lambda x: x[1], reverse=True)

    def _suggest_synonyms(self, target_words, words):
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

    def list_skills(self, job_skills, resume_skills):
        cell_length = max([len(s[0]) for s in job_skills]) + 2
        row_format = "{{:<{}}}".format(cell_length) * 4
        print 'SKILL REPORT for {}'.format(self.job['title'].upper())
        print '************\n'
        print row_format.format('Job Ad Skills', 'Relevance', 'Your Skills', 'Relevance')
        print ''
        for i, skill in enumerate(job_skills):
            if i < len(resume_skills):
                print row_format.format(skill[0].upper(), skill[1], resume_skills[i][0].upper(), resume_skills[i][1])
            else:
                print row_format.format(skill[0].upper(), skill[1], '', '')

    def list_similar_skills(self, target_skill, number_to_return=3):
        similar_skills = self._find_similar_skills(target_skill, [rs[0] for rs in self.resume['skills']])
        for s in similar_skills[:min(number_to_return, len(similar_skills))]:
            print 'Can you modify {} to use {}?'.format(s[0].upper(), target_skill.upper())

    def match_skills(self):
        all_skills = get_all_skills()
        rake = keyword_extractor.KeywordExtractor()
        resume_skills = [s for s in rake.extract(str(self.resume['content']), incl_scores=True) if s[0] in all_skills]
        job_skills = [s for s in rake.extract(str(self.job['description']), incl_scores=True) if s[0] in all_skills]
        matched_skills = [s for s in job_skills if s[0] in [rs[0] for rs in resume_skills]]
        cell_length = max([len(s[0]) for s in job_skills]) + 2
        row_format = "{{:<{}}}".format(cell_length) * 3
        print '\nMATCHING SKILLS'
        print '***************\n'
        print row_format.format('Matching Skill', 'Job Relevance', 'Your Relevance')
        for i, skill in enumerate(matched_skills):
            resume_importance = [rs[1] for rs in resume_skills if rs[0] == skill[0]][0]
            print row_format.format(skill[0].upper(), skill[1], resume_importance)

    def optimize_skills(self):
        # Print out skills in order of relevance
        self.list_skills(self.job['skills'], self.resume['skills'])

        # For the top job skills not in the resume, print out the most similar resume skills
        top_missing_skills = [s[0] for s in self.job['skills'] if s[0] not in [rs[0] for rs in self.resume['skills']]][:5]
        print '\nMISSING SKILL SUGGESTIONS'
        print '*************************\n'
        for skill in top_missing_skills:
            self.list_similar_skills(skill)

    def optimize_acronyms(self):
        # List acronyms
        print '\nACRONYMS TO USE'
        print '****************\n'
        for a in [a[0] for a in self.job['acronyms']]:
            print a

    def optimize_action_words(self):
        print '\nACTION WORD SUGGESTIONS'
        print '***********************\n'
        suggestions = self._suggest_synonyms(self.job['actions'], self.resume['actions'])
        for suggestion in suggestions:
            print 'Can you modify {} to use {}?'.format(suggestion[0].upper(), suggestion[1].upper())

def main(job_file_path, resume_path=None):
    """Analyzes a job description and resume and prints a report. Requires two command line arguments.

    Args:
        job_file_path (str): path to job description text file
        resume_path (str): path to resume text file
    """
    ro = ResumeOptimizer(job_file_path, resume_path)

    # Print similarity
    print 'Similarity: {}\n'.format(ro.similarity)

    # Print skill report
    ro.optimize_skills()

    # Print skills that matched
    ro.match_skills()

    # Print acronyms
    ro.optimize_acronyms()

    # Suggest action word substitutions
    ro.optimize_action_words()

if __name__ == "__main__":
    main(*sys.argv[1:])