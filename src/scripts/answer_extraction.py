from src.data.utils import *
from src.data.data_format import *
import stanza


def extract_answers_from_contexts(qca: QuestionContextAnswer, stanza_dir: str) -> QuestionContextAnswer:

    questions = qca.questions
    nlp = stanza.Pipeline("en", processors='tokenize,ner,pos', use_gpu=True, dir=stanza_dir)
    print('..... Start extracting answers from contexts')

    for q in tqdm(questions):
        q.predicted_answers = q.predicted_answers or []
        existing_answers = []
        for c in q.retrieved_contexts:
            if c.text:
                doc = nlp(c.text)
                num_words = doc.num_words
                if num_words > 15:
                    for ent in doc.ents:
                        ent_text = ent.text
                        ent_type = ent.label_ if hasattr(ent, 'label_') else ent.type
                        if ent.text not in existing_answers:
                            existing_answers.append(ent.text)
                            answer_item = Answer(text=ent_text,
                                                 context=c,
                                                 meta={'ent_type': ent_type},
                                                 start_char_position=ent.start_char,
                                                 end_char_position=ent.end_char)
                            q.predicted_answers.append(answer_item)

    print('...... Filtering out questions without predicted answers')
    new_questions = []
    for q in questions:
        if q.predicted_answers and len(q.predicted_answers) > 0:
            new_questions.append(q)

    new_qca = QuestionContextAnswer(questions=new_questions)
    print(f'Finished extracting answers from contexts! (Number of questions: {len(questions)})')
    return new_qca
