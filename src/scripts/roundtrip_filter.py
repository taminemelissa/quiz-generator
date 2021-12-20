from strsimpy.levenshtein import Levenshtein
from tqdm import tqdm
from transformers import pipeline
from math import inf
import re

from src.data.data_format import *





qa_pipeline = pipeline(
    "question-answering",
    model="csarron/roberta-base-squad-v1",
    tokenizer="csarron/roberta-base-squad-v1"
)



def normalize(s: str) -> str:
    """[summary]

    Args:
        s (str): string to be normalized, i.e. lowered and without punctuation

    Returns:
        str: normalized string
    """
    expression_punctuation = re.compile('[^\w\d\s]')
        
    def remove_punctuation(s: str) -> str:
        
        return re.sub(expression_punctuation, '', s)
    
    return remove_punctuation(s.lower())



def roundtrip_filter(qca: QuestionContextAnswer, threshold: int = 5) -> QuestionContextAnswer:
    """[summary]

    Args:
        qca (QuestionContextAnswer): QuestionContextAnswer instance that we want to filter

    Returns:
        QuestionContextAnswer: QuestionContextAnswer instance containing only contexts and answers that are 'correct',
        i.e. whose answer is not to far from the one found by the BERT model in the sense of the Levenshtein distance
    """
    
    levenshtein = Levenshtein()
    new_questions = []
    
    for q in tqdm(qca.questions):
        
        context = ' '.join([context.text for context in q.retrieved_contexts])
        prediction = qa_pipeline({
            'context': context,
            'question': q.text
        })
        bert_answer = prediction['answer']
        
        
        min_levenshtein_distance = inf
        
        for predicted_answer in tqdm(q.predicted_answers): # we find the best answer in the sens of the Levenshtein distance

            if levenshtein.distance(normalize(predicted_answer.text), normalize(bert_answer)) < min_levenshtein_distance:
                
                best_answer = predicted_answer
                min_levenshtein_distance = levenshtein.distance(normalize(predicted_answer.text), normalize(bert_answer))
                
        if min_levenshtein_distance < threshold:
            
            q.predicted_answers = [best_answer]
            new_questions.append(q)

    new_qca = QuestionContextAnswer(questions = new_questions)
    
    return new_qca