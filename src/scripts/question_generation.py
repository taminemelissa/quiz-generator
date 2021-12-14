from src.models.mt5_generator import MT5Generator
from src.data.data_format import *
from src.data.utils import *


def generate_questions(qca: QuestionContextAnswer, batch_size, model_path) -> QuestionContextAnswer:
    """
    Returns a QuestionContextAnswer object filled with Question/Context/Answer objects from a QuestionContextAnswer
    object filled with Context/Answer objects (generation of question from context and answers)
    :param qca: a QuestionContextAnswer object
    :param batch_size: the batch size for the breaking down into batches of the QuestionContextAnswer object
    :param model_path: the path of the model
    :return: a QuestionContextAnswer object filled with questions
    """

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
