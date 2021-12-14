from src.models.mt5_generator import MT5Generator
from src.data.data_format import *
from src.data.utils import *


def generate_questions(qca: QuestionContextAnswer, batch_size, model_path) -> QuestionContextAnswer:

    device = "cpu"
    print('Loading...')
    generator = MT5Generator(model_path=model_path)
    generator.eval()
    generator.freeze()
    generator.to(device)
    print('Device: ', generator.device)

    new_questions = []
    print('...... Generating questions')
    for question_batch in yield_batches(qca.questions, batch_size=batch_size):
        new_questions += generator.generate(question_batch)

    for q in new_questions:
        q.retrieved_contexts = q.get_all_contexts()

    new_qca = QuestionContextAnswer(questions=new_questions, meta=qca.meta)
    del generator

    print(f'Finished generating questions: (num questions: {len(new_questions)})')
    return new_qca
