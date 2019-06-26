# Copyrights chatme.ai
#   Author: Anna Kozlova
#   Created: 29/04/2019

from collections import namedtuple
import re
from typing import List

from nltk.tokenize import word_tokenize


Token = namedtuple('Token', ('value', 'start', 'end'))


class NLTKSplitter:
    """Nltk Splitter

    This component splits text into tokens using tokenizer from 'nltk' library.
    """

    # nltk_data_path = Path(CONFIG['NLTK_DATA'])

    __DOUBLE_QUOTES_REGEX = re.compile(r'("|\'\'|``)')

    __NLTK_DOUBLE_QUOTES = {
        '``',
        '\'\'',
    }

    __REQUIRED_NLTK_RESOURCES = ['punkt', 'perluniprops']

    def process(self, text: str) -> List[str]:
        """

        :param text: text (already preprocessed if needed)
        :return:
        """

        raw_double_quotes_iterator = self.__DOUBLE_QUOTES_REGEX.finditer(text)

        split_text: List[str] = word_tokenize(text)

        tokens = list()

        for word in split_text:
            if word in self.__NLTK_DOUBLE_QUOTES:
                word = next(raw_double_quotes_iterator).group()

            start = text.find(word, 0 if len(tokens) == 0 else tokens[-1].end)

            tokens.append(Token(
                value=word,
                start=start,
                end=start + len(word),
            ))

        return tokens
