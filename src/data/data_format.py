from typing import Dict, List, Any, OrderedDict, Union
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
                 meta: Dict[str, Any] = None):
        self.text = text
        self.identifier = identifier
        self.scores = scores
        self.meta = meta


class Context(BaseItem):
    def __init__(self, text: str = None,
                 title: str = None,
                 identifier: str = None,
                 scores: OrderedDict[str, float] = None,
                 meta: Dict[str, Any] = None):
        """
        :param text: the text of the context
        :param identifier: the identifier of the context
        :param scores: the score dictionary of the context
        :param meta: the meta information
        """
        super(Context, self).__init__(text=text, identifier=identifier, scores=scores, meta=meta)
        self.title = title


class Answer(BaseItem):
    def __init__(self, text: str = None,
                 context: Context = None,
                 identifier: str = None,
                 scores: OrderedDict[str, float] = None,
                 meta: Dict[str, Any] = None,
                 start_char_position: int = -1,
                 end_char_position: int = -1
                 ):
        """
        :param text: the text of the answer
        :param context: the context from which the answer is extracted
        :param identifier: the identifier of the context
        :param scores: the score dictionary of the context
        :param meta: the meta information
        :param start_char_position : the position of the first character of the answer in the context
        :param end_char_position: the position of the last character of the answer in the context
        """
        super(Answer, self).__init__(text=text, identifier=identifier, scores=scores, meta=meta)
        self.context = context
        self.start_char_position = start_char_position
        self.end_char_position = end_char_position


class Question(BaseItem):
    def __init__(self, text: str = None,
                 identifier: str = None,
                 scores: OrderedDict[str, float] = None,
                 meta: Dict[str, Any] = None,
                 retrieved_contexts: List[Context] = None,
                 gold_answers: List[Answer] = None,
                 predicted_answers: List[Answer] = None
                 ):
        """
        :param text: the text of the question
        :param identifier: the identifier of the question
        :param scores: the score dictionary of the question
        :param retrieved_contexts: the retrieved documents as contexts for this question, if a retriever is involved
        :param meta: the meta information
        :param gold_answers: the gold answers of the question
        :param predicted_answers: the predicted answers of the question
        """
        super(Question, self).__init__(text=text, identifier=identifier, scores=scores, meta=meta)
        self.gold_answers = gold_answers
        self.retrieved_contexts = retrieved_contexts
        self.predicted_answers = predicted_answers

    @staticmethod
    def _get_contexts(answers) -> List[Context]:
        context_ids = []
        contexts = []
        if answers:
            for a in answers:
                if a.context and a.context.identifier not in context_ids:
                    context_ids.append(a.context.identifier)
                    contexts.append(a.context)
        return contexts

    def get_gold_contexts(self) -> List[Context]:
        return self._get_contexts(self.gold_answers)

    def get_predicted_contexts(self) -> List[Context]:
        return self._get_contexts(self.predicted_answers)

    def get_all_contexts(self) -> List[Context]:
        context_ids = []
        contexts = []
        gold_contexts = self.get_gold_contexts()
        pred_contexts = self.get_predicted_contexts()
        retrieved_contexts = self.retrieved_contexts if self.retrieved_contexts else []
        all_contexts = gold_contexts + pred_contexts + retrieved_contexts
        for c in all_contexts:
            if c.identifier not in context_ids:
                context_ids.append(c.identifier)
                contexts.append(c)
        return contexts

    def get_all_answers(self) -> List[Answer]:
        all_answers = []
        if self.gold_answers:
            all_answers += self.gold_answers
        if self.predicted_answers:
            all_answers += self.predicted_answers
        return all_answers


class QuestionContextAnswer:
    def __init__(self, questions: List[Question] = None,
                 meta: Dict[str, Any] = None):
        """
        :param questions: a list of Question objects
        :param meta: the meta information
        """
        self.questions = questions
        self.meta = meta

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        return self.questions[index]

    def get_all_answers(self):
        all_answers = []
        for q in self.questions:
            all_answers += q.get_all_answers()
        return all_answers
