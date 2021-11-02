from typing import Dict, List, Any, OrderedDict, Union
import numpy as np
import collections
import json
from tqdm import tqdm
import random


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

    def get_last_score(self) -> Union[float, None]:
        if not self.scores:
            return None
        return next(reversed(list(self.scores.values())))

    def has_label(self, label: str) -> bool:
        return self.meta and 'labels' in self.meta and label in self.meta['labels']

    def add_label(self, label):
        if not self.meta:
            self.meta = {}
        if not self.meta.get('labels'):
            self.meta['labels'] = []
        if label not in self.meta['labels']:
            self.meta['labels'].append(label)

    def to_dict(self) -> Dict:
        res = {}
        if self.text:
            res.update({
                'text': self.text
            })
        if self.identifier:
            res.update({
                'id': self.identifier,
            })
        if self.meta:
            res.update({
                'meta': self.meta
            })
        if self.scores:
            res.update({
                'scores': dict(self.scores)
            })
        return res

    def __str__(self):
        return self.text


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

    @classmethod
    def from_dict(cls, d: Dict) -> 'Context':
        identifier = d.get('id')
        text = d.get('text')
        title = d.get('title') or d.get('meta', {}).get('title')
        scores = ordered_dict(d.get('scores'))
        meta = d.get('meta')
        return cls(text=text, title=title, identifier=identifier, scores=scores, meta=meta)

    def to_dict(self) -> Dict:
        res = super(Context, self).to_dict()
        if self.title:
            res.update({
                'title': self.title
            })
        return res


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

    @classmethod
    def from_dict(cls, d: Dict, contexts: Dict) -> 'Answer':
        text = d.get('text')
        identifier = d.get('id')
        scores = ordered_dict(d.get('scores'))
        meta = d.get('meta')
        start_char_pos = d.get('start_char_position', -1)
        end_char_pos = d.get('end_char_position', -1)
        start_token_pos = d.get('start_token_position', -1)
        end_token_pos = d.get('end_token_position', -1)
        context_id = d.get('context_id')
        context = Context.from_dict(contexts[context_id]) if context_id else None
        return cls(text=text, identifier=identifier, context=context, scores=scores,
                   meta=meta, start_char_position=start_char_pos, end_char_position=end_char_pos,
                   start_token_position=start_token_pos, end_token_position=end_token_pos)

    def to_dict(self) -> Dict:
        res = super().to_dict()
        if self.context:
            res.update({
                'context_id': self.context.identifier
            })
        if self.start_char_position is not None and self.end_char_position is not None \
                and 0 <= self.start_char_position <= self.end_char_position:
            res.update({
                'start_char_position': self.start_char_position,
                'end_char_position': self.end_char_position
            })
        if self.start_token_position is not None and self.end_token_position is not None \
                and 0 <= self.start_token_position <= self.end_token_position:
            res.update({
                'start_token_position': self.start_token_position,
                'end_token_position': self.end_token_position
            })
        return res


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

    @staticmethod
    def get_contexts(answers: List[Answer]) -> List[Context]:
        context_ids = []
        contexts = []
        if answers:
            for a in answers:
                if a.context and a.context.identifier not in context_ids:
                    context_ids.append(a.context.identifier)
                    contexts.append(a.context)
        return contexts

    @staticmethod
    def get_context_by_identifier(contexts: List[Context], identifier: str) -> Context:
        for context in contexts:
            if context.identifier == identifier:
                return context

    def get_gold_contexts(self) -> List[Context]:
        return self.get_contexts(self.gold_answers)

    def get_predicted_contexts(self) -> List[Context]:
        return self.get_contexts(self.predicted_answers)

    def get_all_answers(self) -> List[Answer]:
        all_answers = []
        if self.gold_answers:
            all_answers += self.gold_answers
        if self.predicted_answers:
            all_answers += self.predicted_answers
        return all_answers

    def get_all_contexts(self) -> List[Context]:
        context_ids = []
        contexts = []
        gold_contexts = self.get_gold_contexts()
        predicted_contexts = self.get_predicted_contexts()
        all_contexts = gold_contexts + predicted_contexts
        for c in all_contexts:
            if c.identifier not in context_ids:
                context_ids.append(c.identifier)
                contexts.append(c)
        return contexts

    def get_gold_answer_texts(self) -> List[str]:
        return [gold_answer.text for gold_answer in self.gold_answers] if self.gold_answers else []

    def get_top_k_answers(self, k: Union[int, None] = 0, pipeline_component: str = None) -> List[Answer]:
        if not self.predicted_answers:
            return []
        predicted_answers = sorted(
            self.predicted_answers,
            key=lambda answer: (answer.get_pipeline_component_score(pipeline_component) is not None,
                                answer.get_pipeline_component_score(pipeline_component)),
            reverse=True)
        return predicted_answers[:min((len(predicted_answers) if k is None else k) + 1, len(predicted_answers))]

    def get_scored_predicted_answers(self) -> Union[None, List[str]]:
        if not self.predicted_answers:
            return None
        return list(self.predicted_answers[0].scores.keys())

    @classmethod
    def from_dict(cls, d: Dict, contexts: Dict) -> 'Question':
        text = d.get('text')
        identifier = d.get('id')
        scores = ordered_dict(d.get('scores'))
        meta = d.get('meta')
        gold_answers = d.get('gold_answers')
        predicted_answers = d.get('predicted_answers')

        if gold_answers:
            gold_answers = [Answer.from_dict(a, contexts) for a in gold_answers]
        if predicted_answers:
            predicted_answers = [Answer.from_dict(a, contexts) for a in predicted_answers]

        return cls(text=text, identifier=identifier, scores=scores, meta=meta,
                   gold_answers=gold_answers,
                   predicted_answers=predicted_answers)

    def to_dict(self) -> Dict:
        res = super().to_dict()
        if self.gold_answers:
            res.update({
                'gold_answers': [a.to_dict() for a in self.gold_answers]
            })
        if self.predicted_answers:
            res.update({
                'predicted_answers': [a.to_dict() for a in self.predicted_answers]
            })
        return res

    def print_top_k_answers(self, k: int):
        print('===' * 50)
        print(f'Question: {self.text}')
        for predicted_answer in self.get_top_k_answers(k):
            print(f'--' * 50)
            print(f'Answer: {predicted_answer.text}')
            print(f'Document title: {predicted_answer.context.meta.get("document_title")}')


class QuestionCollection:
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

    @classmethod
    def from_dict(cls, d) -> 'QuestionCollection':
        """
        :param d: the dictionary object loaded from a json file with the following predefined format
            {
                "contexts": [ {"context_text":"...", "context_id":"c_id", "meta": {}}, ... ],
                "qas": [ {
                           "question_id": "q_id",
                           "question_text: "...",
                           "gold_answers": [{
                                "answer_text":"...",
                                "context_id":"c_id",
                                "start_char_position":30,
                                "end_char_position":31.
                                "meta": {}
                           }],
                           "predicted_answers": [],
                           "scores": {
                                "a_score": 1.23,
                                "b_score": 4.56
                            },
                           "meta": {}
                        } ],
                "meta": {}
            }
        """
        contexts = d['contexts']
        qas = d['qas']
        questions = [Question.from_dict(qa, contexts) for qa in qas]
        return cls(questions=questions, meta=d.get('meta'), scores=d.get('scores'))

    def to_dict(self) -> Dict:
        res = {}
        if self.questions:
            contexts = {}
            qas = []
            for q in self.questions:
                qas.append(q.to_dict())
                q_contexts = q.get_all_contexts()
                for c in q_contexts:
                    if c.identifier and c.identifier not in contexts:
                        contexts[c.identifier] = c.to_dict()
            res.update({
                'contexts': contexts,
                'qas': qas
            })
        if self.meta:
            res.update({
                'meta': self.meta
            })
        if self.scores:
            res.update({
                'scores': self.scores
            })
        return res

    def save(self, output_path: str, indent=None):
        print(f'Save question collection to {output_path}')
        with open(output_path, 'w', encoding='utf8') as out:
            json.dump(self.to_dict(), out, ensure_ascii=False, indent=indent)

    @classmethod
    def load(cls, input_path: str) -> 'QuestionCollection':
        print(f'Load question collection from {input_path}')
        with open(input_path, 'r', encoding='utf8') as f:
            json_obj = json.load(f)
            return cls.from_dict(json_obj)

    def batch(self, batch_size=12) -> List[Question]:
        for i in tqdm(range(0, len(self.questions), batch_size)):
            yield self.questions[i: i + batch_size]

    def random_subset(self, k: int) -> List[Question]:
        indices = random.choices(range(0, len(self.questions)), k=k)
        result = []
        for i in indices:
            result.append(self.questions[i])
        return result

    def get_all_answers(self):
        all_answers = []
        for q in self.questions:
            all_answers += q.get_all_answers()
        return all_answers

    def get_all_contexts(self):
        all_contexts = []
        for q in self.questions:
            all_contexts += q.get_all_contexts()
        return all_contexts

    def print_stats(self):
        def print_array_stats(arr: List, prefix: str = '', decimal: int = 1) -> str:
            if arr:
                result_line = f'{prefix} ({len(arr)}): ' \
                              f'{round(float(np.mean(arr)), decimal)} ' \
                              f'(min: {round(float(np.min(arr)), decimal)}, ' \
                              f'max: {round(float(np.max(arr)), decimal)}, ' \
                              f'sd: {round(float(np.std(arr)), decimal)}, ' \
                              f'median: {round(float(np.median(arr)), decimal)})'
                print(result_line)
                return result_line
            else:
                print(f'{prefix}: NA')
                return f'{prefix}: NA'

        question_lengths = []
        num_gold_answers_per_question = []
        num_predicted_answers_per_question = []
        gold_answer_lengths = []
        predicted_answer_lengths = []
        gold_context_lengths = []
        predicted_context_lengths = []
        for q in tqdm(self.questions):
            question_lengths.append(len(q.text.split()))
            if q.gold_answers:
                num_gold_answers_per_question.append(len(q.gold_answers))
                for a in q.gold_answers:
                    gold_answer_lengths.append(len(a.text.split()))
                    gold_context_lengths.append(len(a.context.text.split()))
            if q.predicted_answers:
                num_predicted_answers_per_question.append(len(q.predicted_answers))
                for a in q.predicted_answers:
                    predicted_answer_lengths.append(len(a.text.split()))
                    predicted_context_lengths.append(len(a.context.text.split()))

        print('-----------------------------------------------------------')
        print(f'Questions: {len(self.questions)}')
        print_array_stats(question_lengths, 'Question length')
        print_array_stats(num_gold_answers_per_question, 'Gold answers per question')
        print_array_stats(gold_answer_lengths, 'Gold answer length')
        print_array_stats(gold_context_lengths, 'Gold context length')
        print_array_stats(num_predicted_answers_per_question, 'Predicted answers per question')
        print_array_stats(predicted_answer_lengths, 'Predicted answer length')
        print_array_stats(predicted_context_lengths, 'Predicted context length')
        print('-----------------------------------------------------------')
