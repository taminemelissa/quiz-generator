from typing import Dict, List, Any, OrderedDict, Union
import numpy as np
import collections


def ordered_dict(obj: dict) -> Union[OrderedDict, object]:
    if type(obj) == 'dict':
        return collections.OrderedDict(obj)
    else:
        return obj


class BaseItem(object):
    def __init__(self, text: str = None,
                 identifier: str = None,
                 scores: OrderedDict[str, float] = None,
                 vector: np.ndarray = None,
                 meta: Dict[str, Any] = None):
        self.text = text
        self.identifier = identifier
        self.scores = scores
        self.vector = vector
        self.meta = meta


class Context(BaseItem):
    def __init__(self, text: str = None,
                 title: str = None,
                 identifier: str = None,
                 scores: OrderedDict[str, float] = None,
                 vector: np.ndarray = None,
                 meta: Dict[str, Any] = None):
        """
        :param text: the text of the context
        :param identifier: the identifier of the context
        :param scores: the score dictionary of the context
        :param vector: the vector representation of the context
        :param meta: the meta information
        """
        super(Context, self).__init__(text=text, identifier=identifier, scores=scores, vector=vector, meta=meta)
        self.title = title


class Answer(BaseItem):
    def __init__(self, text: str = None,
                 context: Context = None,
                 identifier: str = None,
                 scores: OrderedDict[str, float] = None,
                 vector: np.ndarray = None,
                 meta: Dict[str, Any] = None,
                 start_char_position: int = -1,
                 end_char_position: int = -1,
                 start_token_position: int = -1,
                 end_token_position: int = -1
                 ):
        """
        :param text: the text of the answer
        :param context: the context from which the answer is extracted
        :param identifier: the identifier of the context
        :param scores: the score dictionary of the context
        :param vector: the vector representation of the context
        :param meta: the meta information
        :param start_char_position: the position of the first character of the answer into the context
        :param end_char_position: the position of the last character of the answer into the context
        :param start_token_position: the position of the first token of the answer into the (tokenized) context
        :param end_token_position: the position of the last token of the answer into the (tokenized) context
        """
        super(Answer, self).__init__(text=text, identifier=identifier, scores=scores, vector=vector, meta=meta)
        self.start_char_position = start_char_position
        self.end_char_position = end_char_position
        self.start_token_position = start_token_position
        self.end_token_position = end_token_position
        self.context = context


class Question(BaseItem):
    def __init__(self, text: str = None,
                 identifier: str = None,
                 scores: OrderedDict[str, float] = None,
                 vector: np.ndarray = None,
                 meta: Dict[str, Any] = None,
                 gold_answers: List[Answer] = None,
                 predicted_answers: List[Answer] = None
                 ):
        """
        :param text: the text of the question
        :param identifier: the identifier of the question
        :param scores: the score dict of the question
        :param vector: the vector representation of the question
        :param meta: the meta information
        :param gold_answers: the gold answers of the question
        :param predicted_answers: the predicted answers of the question
        """
        super(Question, self).__init__(text=text, identifier=identifier, scores=scores, vector=vector, meta=meta)
        self.gold_answers = gold_answers
        self.predicted_answers = predicted_answers


class QuestionContextAnswer:
    def __init__(self, questions: List[Question] = None,
                 meta: Dict[str, Any] = None,
                 scores: Dict = None):
        self.questions = questions
        self.meta = meta
        self.scores = ordered_dict(scores or {})

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        return self.questions[index]