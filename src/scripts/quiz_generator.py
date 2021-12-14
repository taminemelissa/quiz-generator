import random
from src.data.data_format import *
from src.models.bm25_retriever import BM25Retriever
from src.scripts.answer_extraction import extract_answers_from_contexts
from src.scripts.question_generation import generate_questions


def quiz_generator(theme: str):
    """
    Generates a quiz composed of 10 questions/answers pairs about a given theme
    :param theme: the theme of the quiz
    """

    bm25 = BM25Retriever()
    contexts = bm25.retrieve(query=theme)
    questions = [Question(retrieved_contexts=[context]) for context in contexts]
    qca = QuestionContextAnswer(questions=questions)

    qca = extract_answers_from_contexts(qca, '')
    qca = generate_questions(qca, 12, model_path="Narrativa/mT5-base-finetuned-tydiQA-question-generation")

    displayed_questions = random.choices(qca.questions, k=10)
    for i in range(len(displayed_questions)):
        print(i,".  Question : ", displayed_questions[i].text, "?  Réponse : ", displayed_questions[i].predicted_answers[0].text)


