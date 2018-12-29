from typing import List

import re
import numpy as np
import pandas as pd

from eunjeon import Mecab # Uses mecab for better performance
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = ['ColumnSelector', 'ColumnMerger', 'WordUnifier',
           'RegExReplacer', 'DuplicateRemover', 'StopWordRemover',
           'WordLower', 'MorphTokenizer', 'NounTokenizer', 'PosTokenizer']


############################
# 1. DataFrame Preprocessing
#    - ColumnSelector
#    - ColumnMerger
############################


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    주어진 데이터프레임에서 Pipeline에서 적용할 컬럼을 선택

    Example

    >>> df = pd.DataFrame(data={ "과일" : ['사과','배','딸기'],"시장" : ['명동','상정','죽도']})
    >>> cs = ColumnSelector("과일")
    >>> cs.transform(df)
    0    사과
    1     배
    2    딸기
    Name: 과일, dtype: object

    """

    def __init__(self, col_name):
        self.col_name = col_name

    def fit(self, X, y=None):
        if self.col_name not in X.columns:
            raise ValueError("DataFrame 내에 {}가 없습니다.".format(self.col_name))
        return self

    def transform(self, X):
        return X[self.col_name]


class ColumnMerger(BaseEstimator, TransformerMixin):
    """
    주어진 데이터프레임에서 컬럼에 해당하는 string을 합치는

    Example

    >>> df = pd.DataFrame(data={ "과일" : ['사과','배','딸기'],"시장" : ['명동','상정','죽도']})
    >>> cs = ColumnMerger(['과일','시장'])
    >>> cs.transform(df)
    0    사과 명동
    1     배 상정
    2    딸기 죽도
    dtype: object

    """

    def __init__(self, col_names=[]):
        self.col_names = col_names

    def fit(self, X, y=None):
        for col_name in self.col_names:
            if col_name not in X.columns:
                raise ValueError("DataFrame 내에 {}가 없습니다.".format(col_name))
        return self

    def transform(self, X):
        return X[self.col_names].apply(lambda x: " ".join(x), axis=1)


############################
# 2. Basic NLP Preprocssing
#    - WordUnifier
#
#    - DuplicateRemover
#    - StopWordRemover
#    - RegExReplacer
#
#    - WordLower
############################
class WordUnifier(BaseEstimator, TransformerMixin):
    """
    동일의미 다른 표기 통일

    # TODO : 구현은 쉽지만, 잘못 구현 할 경우 속도 이슈가 날 거 같습니다.
    # 속도 이슈 없는 코드를 원합니다!

    Example

    >>> sample = np.array(['삼성전자 노트북', "노트북 삼성", "samsung 스마트폰", 'lg 폰', "엘지전자 상거래"])
    >>> wu = WordUnifier([["삼성","삼성전자",'samsung'], ["엘지",'엘지전자','lg']])
    >>> wu.transform(sample)
    array(['삼성 노트북', "노트북 삼성", "삼성 스마트폰", '엘지 폰', "엘지 상거래"], dtype=object)

    """

    def __init__(self, words_list=[]):
        self._words_list = words_list

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            x_shape = X.shape
            return np.array([self._transform(phrase)
                             for phrase in X.ravel()]).reshape(x_shape)
        elif isinstance(X, pd.Series):
            return X.map(self._transform)
        elif isinstance(X, pd.DataFrame):
            return X.applymap(self._transform)
        elif isinstance(X, list) or isinstance(X, tuple):
            return [self._transform(phrase) for phrase in X]
        else:
            raise TypeError("적절하지 못한 DataType이 들어왔습니다.")

    @staticmethod
    def _transform(phrase):
        # TODO : wordunifier 구현
        return


class RegExReplacer(BaseEstimator, TransformerMixin):
    """
    정규식을 활용한 word 치환
    주어진 정규식에 만족하는 word에 대해서, 특정 word로 변경하는 코드

    Example

    >>>
    >>>
    >>>
    """

    def __init__(self, regex_list=[]):
        self._regex_list = regex_list

    def fit(self, X, y=None):
        return X

    def transform(self, X):
        if isinstance(X, np.ndarray):
            x_shape = X.shape
            return np.array([self._transform(phrase)
                             for phrase in X.ravel()]).reshape(x_shape)
        elif isinstance(X, pd.Series):
            return X.map(self._transform)
        elif isinstance(X, pd.DataFrame):
            return X.applymap(self._transform)
        elif isinstance(X, list) or isinstance(X, tuple):
            return [self._transform(phrase) for phrase in X]
        else:
            raise TypeError("적절하지 못한 DataType이 들어왔습니다.")

    @staticmethod
    def _transform(phrase) -> List:
        if re.search(r'[0-9]+(kg|KG|Kg)', phrase) is not None:
            result = re.sub(r'[0-9]+(kg|KG|Kg)', '<단위>', phrase)
        elif re.search(r'[0-9]+.(L)', phrase) is not None:
            result = re.sub(r'[0-9]+(L)', '<부피단위>', phrase)
        else:
            result = phrase
        return result


class DuplicateRemover(BaseEstimator, TransformerMixin):
    """
    중복 단어 제거

    Example

    >>> sample = np.array(['청동 사과 할인 특가 사과', "삼성 컴퓨터 특가 세일 삼성", "완전 싸다 완전 초대박 싸다"])
    >>> dr = DuplicateRemover()
    >>> dr.transform(sample)
    array(['청동 사과 할인 특가', '삼성 컴퓨터 특가 세일', '완전 싸다 초대박'], dtype='<U12')
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            x_shape = X.shape
            return np.array([self._transform(phrase)
                             for phrase in X.ravel()]).reshape(x_shape)
        elif isinstance(X, pd.Series):
            return X.map(self._transform)
        elif isinstance(X, pd.DataFrame):
            return X.applymap(self._transform)
        elif isinstance(X, list) or isinstance(X, tuple):
            return [self._transform(phrase) for phrase in X]
        else:
            raise TypeError("적절하지 못한 DataType이 들어왔습니다.")

    @staticmethod
    def _transform(phrase):
        return " ".join(list(dict.fromkeys(phrase.split(" "))))


class StopWordRemover(BaseEstimator, TransformerMixin):
    """
    불용어를 제거

    Example
    >>> sample = ["노트북 할인 판매", "옷 기타 완전 세일", "비아그라 할인", "클래식기타 판매 세일", "판매왕의 판매"]
    >>> transformer = StopWordRemover(['판매', '기타'])
    >>> transformer.transform(sample)
    ["노트북 할인", "옷 완전 세일", "비아그라 할인", "클래식기타 세일", "판매왕의"]
        pred = transformer.transform(answer)
    """

    def __init__(self, stop_words=[]):
        self._stop_words = stop_words
        self._sw_regex = re.compile(r'\b%s\b' %
                                    r'\b|\b'.join(map(re.escape, self._stop_words)))
        self._ds_regex = re.compile(r"\s+")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            x_shape = X.shape
            return np.array([self._transform(phrase)
                             for phrase in X.ravel()]).reshape(x_shape)
        elif isinstance(X, pd.Series):
            return X.map(self._transform)
        elif isinstance(X, pd.DataFrame):
            return X.applymap(self._transform)
        elif isinstance(X, list) or isinstance(X, tuple):
            return [self._transform(phrase) for phrase in X]
        else:
            raise TypeError("적절하지 못한 DataType이 들어왔습니다.")

    def _transform(self, phrase):
        _phrase = self._sw_regex.sub("", phrase)
        return self._ds_regex.sub(" ", _phrase).strip()


class WordLower(BaseEstimator, TransformerMixin):
    """
    모두 소문자화

    >>> sample = np.array(['Kang', "KAM", "Kan"])
    >>> wl = WordLower()
    >>> wl.transform(sample)
    array(['kang', 'kam', 'kan'], dtype='<U4')

    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            x_shape = X.shape
            return np.array([self._transform(phrase)
                             for phrase in X.ravel()]).reshape(x_shape)
        elif isinstance(X, pd.Series):
            return X.map(self._transform)
        elif isinstance(X, pd.DataFrame):
            return X.applymap(self._transform)
        elif isinstance(X, list) or isinstance(X, tuple):
            return [self._transform(phrase) for phrase in X]
        else:
            raise TypeError("적절하지 못한 DataType이 들어왔습니다.")

    @staticmethod
    def _transform(word):
        return word.lower()


############################
# 3. Tokenizer
#   - MorphTokenizer
#   - NounTokenizer
#   - PosTokenizer
# TODO : 이 쪽은 transform 코드를 다 짠후 리팩토링 하려고 합니다.
# 고민포인트
#   konlpy를 wrapping하여 구성하려고 하는데
#   twitter를 주로 사용한다는 가정으로 설계하였습니다.
#   (좋지 못한 가정이고, 코드의 유연성을 떨어트리는 못된 행위이지요)
#   어떤 식으로 확장해야 좀 더 좋은 코드가 될 것인지
#   고민이 좀 들고 있었습니다.
############################
class MorphTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._mecab = Mecab()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            x_shape = X.shape
            return np.array([self._transform(phrase)
                             for phrase in X.ravel()]).reshape(x_shape)
        elif isinstance(X, pd.Series):
            return X.map(self._transform)
        elif isinstance(X, pd.DataFrame):
            return X.applymap(self._transform)
        elif isinstance(X, list) or isinstance(X, tuple):
            return [self._transform(phrase) for phrase in X]
        else:
            raise TypeError("적절하지 못한 DataType이 들어왔습니다.")

    def _transform(self, phrase):
        return " ".join(self._mecab.morphs(phrase))


class NounTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._mecab = Mecab()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            x_shape = X.shape
            return np.array([self._transform(phrase)
                             for phrase in X.ravel()]).reshape(x_shape)
        elif isinstance(X, pd.Series):
            return X.map(self._transform)
        elif isinstance(X, pd.DataFrame):
            return X.applymap(self._transform)
        elif isinstance(X, list) or isinstance(X, tuple):
            return [self._transform(phrase) for phrase in X]
        else:
            raise TypeError("적절하지 못한 DataType이 들어왔습니다.")

    def _transform(self, phrase):
        return " ".join(self._mecab.nouns(phrase))


class PosTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, norm=False, stem=False,
                 excludes=['Punctuation', 'Number', 'Foreign']):
        self._norm = norm
        self._stem = stem
        self._excludes = excludes
        self._mecab = Mecab()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            x_shape = X.shape
            return np.array([self._transform(phrase)
                             for phrase in X.ravel()]).reshape(x_shape)
        elif isinstance(X, pd.Series):
            return X.map(self._transform)
        elif isinstance(X, pd.DataFrame):
            return X.applymap(self._transform)
        elif isinstance(X, list) or isinstance(X, tuple):
            return [self._transform(phrase) for phrase in X]
        else:
            raise TypeError("적절하지 못한 DataType이 들어왔습니다.")

    def _transform(self, phrase):
        pos_list = self._mecab.pos(phrase)
        pos_drop = list(filter(
            lambda pos: pos[1] not in self._excludes, pos_list))

        if len(pos_drop) == 0:
            return ""
        else:
            return " ".join(list(zip(*pos_drop))[0])
