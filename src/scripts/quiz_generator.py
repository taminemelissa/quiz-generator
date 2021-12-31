import random
from src.data.data_format import *
from src.models.bm25_retriever import BM25Retriever
from src.scripts.wikipedia_indexing import set_es_client
from src.scripts.answer_extraction import extract_answers_from_contexts
from src.scripts.question_generation import generate_questions
from src.scripts.roundtrip_filter import roundtrip_filter


def quiz_generator(theme: str):
    """
    Generates a quiz composed of 10 questions/answers pairs about a given theme
    :param theme: the theme of the quiz
    """
    es = set_es_client()
    bm25 = BM25Retriever(client=es)
    contexts = bm25.retrieve(query=theme)
    questions = [Question(retrieved_contexts=[context]) for context in contexts]
    qca = QuestionContextAnswer(questions=questions)

    qca = extract_answers_from_contexts(qca, 'data/stanza')
    qca = generate_questions(qca, 12, model_path="Narrativa/mT5-base-finetuned-tydiQA-question-generation")
    qca = roundtrip_filter(qca, model_path="csarron/roberta-base-squad-v1", threshold = 6)
    
    if len(qca.questions)>10:
        displayed_questions = random.choices(qca.questions, k=10)
    else:
        displayed_questions = qca.questions
    for i in range(len(displayed_questions)):
        print(f'{i+1}.{displayed_questions[i].text}? \ {displayed_questions[i].predicted_answers[0].text}\n')


