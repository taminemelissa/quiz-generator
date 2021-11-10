from data.data_format import *


def split_into_train_test_validation_sets(questions: List[Question],
                                          train_ratio: float = 0.7,
                                          test_ratio: float = 0.2) -> dict:
    total = len(questions)
    train_count = int(train_ratio * total)
    test_count = int(test_ratio * total)
    if total - train_count - test_count > 0:
        random.shuffle(questions)
        train_questions = questions[0:train_count]
        test_questions = questions[train_count:train_count + test_count]
        dev_questions = questions[train_count + test_count:]
        return {
            'train': train_questions,
            'test': test_questions,
            'dev': dev_questions
        }
    else:
        raise Exception('Set correct ratios')
