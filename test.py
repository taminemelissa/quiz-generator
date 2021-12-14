if __name__ == '__main__':
    import os
    import sys

    project_dir = os.getcwd().split('src')[0]
    sys.path.append(project_dir)
    from src.data.data_format import *
    from src.scripts.answer_extraction import *
    from src.scripts.question_generation import *

    contexts = [Context(text="Before forming Queen, May and Taylor had played together in the band Smile. Mercury was a fan of Smile and encouraged them to experiment with more elaborate stage and recording techniques. He joined in 1970 and suggested the name Queen. Deacon was recruited in February 1971, before the band released their eponymous debut album in 1973.")]
    questions = [Question(retrieved_contexts=[context]) for context in contexts]
    qca = QuestionContextAnswer(questions=questions)
    stanza_dir = "data/stanza"
    qca = extract_answers_from_contexts(qca, stanza_dir)
    qca = generate_questions(qca, 12, model_path="Narrativa/mT5-base-finetuned-tydiQA-question-generation")
    print(qca.questions[1].text, qca.questions[1].predicted_answers[0].text)

