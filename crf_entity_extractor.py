# Copyrights chatme.ai
#   Author: Anna Kozlova
#   Created: 17/05/2019

import os
from pathlib import Path
from collections import namedtuple
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Union

from nltk import pos_tag
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn_crfsuite import CRF

from nltk_splitter import NLTKSplitter

__all__ = [
    'Entity',
    'CrfEntityExtractor',
]


Entity = namedtuple('Entity', ['name', 'value', 'start_token', 'end_token', 'start', 'end'])


class CrfEntityExtractor:

    __DIRNAME = Path(os.path.dirname(__file__))

    __FEATURES_SET = [
        ['low', 'title', 'upper', 'pos', 'pos2'],
        ['low', 'word3', 'word2', 'upper', 'title', 'digit', 'pos', 'pos2'],
        ['low', 'title', 'upper', 'pos', 'pos2'],
    ]

    __HALF_FEATURES_SPAN = len(__FEATURES_SET) // 2

    __CONFIG = {
        'max_iterations': 40,
        'L1_c': 1e-3,
        'L2_c': 1e-2,
    }

    __FEATURES_RANGE = range(-__HALF_FEATURES_SPAN, __HALF_FEATURES_SPAN + 1)

    __PREFIXES = [str(i) for i in __FEATURES_RANGE]

    __FUNCTION_DICT = {
        'low': lambda doc: doc[0].lower(),
        'title': lambda doc: doc[0].istitle(),
        'word3': lambda doc: doc[0][-3:],
        'word2': lambda doc: doc[0][-2:],
        'word1': lambda doc: doc[0][-1:],
        'pos': lambda doc: doc[1],
        'pos2': lambda doc: doc[1][:2],
        'bias': lambda doc: 'bias',
        'upper': lambda doc: doc[0].isupper(),
        'digit': lambda doc: doc[0].isdigit(),
    }

    def __init__(self):
        self.__crf_model = None

    def fit(self, train_data: Iterable[str], labels: Iterable[Iterable[str]]):
        """

        :param train_data:
        :param labels: labels in BIO or BILOU notation
        :return:
        """

        crf_dataset = self.__create_dataset(train_data, labels)

        features = [
            self.__convert_idata_to_features(message_data)
            for message_data in crf_dataset
        ]

        labels = [
            self.__extract_labels_from_data(message_data)
            for message_data in crf_dataset
        ]

        self.__crf_model = CRF(
            algorithm='lbfgs',
            c1=self.__CONFIG['L1_c'],
            c2=self.__CONFIG['L2_c'],
            max_iterations=self.__CONFIG['max_iterations'],
            all_possible_transitions=True,
        )

        self.__crf_model.fit(features, labels)

        return self

    def predict(self, text: str) -> List['Entity']:
        """Predicts entities in text.

        :param text:
        :return:
        """

        tokens = self.__preprocess(text)

        intermediate_data = self.__convert_to_idata_format(tokens)
        features = self.__convert_idata_to_features(intermediate_data)

        predicts = self.__crf_model.predict_marginals_single(features)
        
        entities = []
        for pred in predicts:
            sorted_pred = sorted(pred.items(), key=lambda x: x[1], reverse=True)
            entities.append(sorted_pred[0][0])

        # entities = self.__get_entities_from_predict(
        #     tokens,
        #     predicts
        # )

        return entities

    def evaluate(self, test_data: Iterable[str], labels: Iterable[Iterable[str]], metric: str = 'accuracy'):
        """Evaluates accuracy on test data.

        :param test_data:
        :param labels:
        :param metric:
        :return:
        """
        # if self.__crf_model is None:
        #     self.load_model()

        labels = self.__process_test_labels(labels)
        predicted_entities = [self.predict(sentence) for sentence in test_data]
        processed_predicted_entities = [self.__postprocess(sent_entities, self.__preprocess(sentence))
                                        for (sent_entities, sentence) in zip(predicted_entities, test_data)]

        all_predicted = self.__get_flatten_values(processed_predicted_entities)

        all_labels = self.__get_flatten_values(labels)
        return accuracy_score(all_predicted, all_labels)

    def load_model(self, path: Path) -> 'CrfEntityExtractor':
        """Loads saved model.

        :param path: path where model was saved
        :return:
        """

        self.__crf_model = joblib.load(path)

        return self

    def save_model(self, path: Path) -> None:
        joblib.dump(self.__crf_model, path)

    def __create_dataset(self, sentences, labels):
        dataset_message_format = [
            self.__convert_to_idata_format(
                self.__preprocess(sentence),
                sentence_labels
            )
            for sentence, sentence_labels in zip(sentences, labels)
        ]
        return dataset_message_format

    def __convert_to_idata_format(self, tokens, entities=None):
        message_data_intermediate_format = []

        for i, token in enumerate(tokens):
            entity = entities[i] if (entities and len(entities) > i) else "N/A"
            tag = self.__get_tag_of_token(token.value)
            message_data_intermediate_format.append((token.value, tag, entity))

        return message_data_intermediate_format

    def __get_entities_from_predict(self, tokens, predicts):

        entities = []
        cur_token_ind: int = 0

        while cur_token_ind < len(tokens):
            end_ind, confidence, entity_label = self.__handle_bilou_label(
                cur_token_ind,
                predicts
            )

            if end_ind is not None:

                current_tokens = tokens[cur_token_ind: end_ind + 1]

                entity_value: str = ' '.join([token.value for token in current_tokens])
                entity = Entity(
                    name=entity_label,
                    value=entity_value,
                    start_token=cur_token_ind,
                    end_token=end_ind,
                    start=current_tokens[0].start,
                    end=current_tokens[-1].end
                )

                entities.append(entity)
                cur_token_ind = end_ind + 1
            else:
                cur_token_ind += 1

        return entities

    def __handle_bilou_label(self, token_index, predicts):

        label, confidence = self.__get_most_likely_entity(token_index, predicts)
        entity_label = self.__convert_to_ent_name(label)

        extracted = self.__extract_bilou_prefix(label)

        # if extracted == "U":
        #     return token_index, confidence, entity_label

        if extracted == "B":
            end_token_index, confidence = self.__find_bilou_end(
                token_index,
                predicts
            )
            return end_token_index, confidence, entity_label

        else:
            return None, None, None

    def __find_bilou_end(self, token_index: int, predicts):

        end_token_ind: int = token_index + 1
        finished: bool = False

        label, confidence = self.__get_most_likely_entity(token_index, predicts)
        entity_label: str = self.__convert_to_ent_name(label)

        while not finished:
            label, label_confidence = self.__get_most_likely_entity(
                end_token_ind,
                predicts
            )

            confidence = min(confidence, label_confidence)

            if self.__convert_to_ent_name(label) != entity_label:

                if self.__extract_bilou_prefix(label) == 'L':
                    finished = True

                if self.__extract_bilou_prefix(label) == 'I':
                    end_token_ind += 1

                else:
                    finished = True
                    end_token_ind -= 1

            else:
                end_token_ind += 1

        return end_token_ind, confidence


    def __mark_positions_by_labels(self, entities_labels, positions, name: str):

        if len(positions) == 1:
            entities_labels = self.__set_label(
                entities_labels, positions[0], 'U', name
            )
        else:
            entities_labels = self.__set_label(
                entities_labels, positions[0], 'B', name
            )
            entities_labels = self.__set_label(
                entities_labels, positions[-1], 'L', name
            )
            for ind in positions[1: -1]:
                entities_labels = self.__set_label(
                    entities_labels, ind, 'I', name
                )

        return entities_labels

    def __get_example_features(self, data, example_index):
        """Exctract features from example in intermediate data format

        :param data: list of examples in task specified format
        :param example_index: index of central example
        :return: list of special features extracted from one example and its context
        """

        message_len = len(data)
        example_features = {}

        for futures_index in self.__FEATURES_RANGE:

            if example_index + futures_index >= message_len:
                example_features['EOS'] = True
            elif example_index + futures_index < 0:
                example_features['BOS'] = True
            else:
                example = data[example_index + futures_index]
                shifted_futures_index = futures_index + self.__HALF_FEATURES_SPAN
                prefix = self.__PREFIXES[shifted_futures_index]
                features = self.__FEATURES_SET[shifted_futures_index]

                for feature in features:
                    value = self.__FUNCTION_DICT[feature](example)
                    example_features[f'{prefix}:{feature}'] = value

        return example_features

    def __convert_idata_to_features(self, data):
        """Extract features from examples in intermediate data format

        :param data: list of examples in special format
        :return: list of futures extracted form each example
        """

        features = []

        for ind, example in enumerate(data):
            example_features: Dict[str, Any] = self.__get_example_features(data, ind)
            features.append(example_features)

        return features

    def __get_most_likely_entity(self, ind: int, predicts):

        if len(predicts) > ind:
            entity_probs = predicts[ind]
        else:
            entity_probs = None

        if entity_probs:
            label: str = max(entity_probs, key=lambda key: entity_probs[key])
            confidence = sum([v for k, v in entity_probs.items() if k[2:] == label[2:]])

            return label, confidence
        else:
            return '', 0.0

    def __convert_to_ent_name(self, bilou_ent_name: str) -> str:
        """Get entity name from bilou label representation

        :param bilou_ent_name: BILOU entity name
        :return: entity name without BILOU prefix
        """

        return bilou_ent_name[2:]

    def __extract_bilou_prefix(self, label: str):
        """Get BILOU prefix from label

        If label prefix (first label symbol) not in {B, I, U, L} return None

        :param label: BILOU entity name
        :return: BILOU prefix
        """

        if len(label) >= 2 and label[1] == "-":
            return label[0].upper()

        return None

    def __process_test_labels(self, test_labels):
        return [self.__to_dict([label[2:]
                         if (label.startswith('B-') or label.startswith('I-')) else label
                         for label in sent_labels])
                for sent_labels in test_labels]

    @staticmethod
    def __preprocess(text: str) -> List[str]:
        """Deletes EOS token; splits texts into token.

        :param texts:
        :return: tokens
        """

        if text.endswith('EOS'):
            return NLTKSplitter().process(text)[:-1]
        else:
            return NLTKSplitter().process(text)

    @staticmethod
    def __get_tag_of_token(token: str) -> str:
        """Gets part-of-speech tag for token.

        :param token:
        :return: POS tag
        """

        tag = pos_tag([token])[0][1]
        return tag

    @staticmethod
    def __extract_labels_from_data(data: Iterable) -> List[str]:

        return [label for _, _, label in data]

    @staticmethod
    def __set_label(entities_labels, ind: int, prefix: str, name: str):

        entities_labels[ind] = f'{prefix}-{name}'

        return entities_labels

    @staticmethod
    def __to_dict(sent_labels):
        return dict(enumerate(sent_labels))

    @staticmethod
    def __postprocess(entities, sentence):
        entities_dict = {k: 'O' for k in range(len(sentence))}
        for entity in entities:
            for key in range(entity.start_token, entity.end_token + 1):
                entities_dict[key] = entity.name
        return entities_dict

    @staticmethod
    def __get_flatten_values(dicts):

        return [word for sentence in dicts for word in sentence.values()]
