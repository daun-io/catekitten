# Catekitten

Catekitten은 Kakao-arena(1. 쇼핑몰 카테고리 분류) 참가에 사용된 모든 코드와 아이디어를 공개하는 저장소입니다.

## Dependencies

```
python==3.x

h5py
numpy
tensorflow
pandas
fire
eunjeon
sklearn
```

## 모델 재구현

카카오 아레나 홈페이지에서 다운로드받은 데이터를 모두 저장소의 ./data/raw 폴더에 위치시킵니다.

```
# 저장소 clone
git clone https://github.com/nyanye/catekitten.git
cd catekitten

# HDF 전처리
python -m catekitten.data merge "./data/raw/" "./data/prep/textonly.h5" 

# 학습 및 추론 결과 생성, 메모리를 많이 요구할 수 있습니다.
python -m catekitten.train
```

## 방법론

### 사용 Features

#### Text features

* brand  
* product  
* maker  
* model  

### Tokenization

위에서 정의된 Text features는 형태소 분석기 mecab의 한국어 이식 프로젝트 은전한닢 (python 인터페이스로 [eunjeon](https://github.com/koshort/pyeunjeon)을 활용했습니다.)을 통해 단어로 구분되며, keras.preprocessing.text.Tokenizer를 활용해 1개의 데이터에 대한 1개의 feature (ex: product)는 최대길이 32를 가진 sequence 행렬로 변환됩니다. (4종류의 feature를 활용하므로 총 길이 128의 sequence 행렬로 변환됨)

### 네트워크 구조

김윤님의 문장 분류에 관한 논문 및 Apache 2.0 라이센스로 공개된 [카카오의 khaii 저장소](https://github.com/kakao/khaiii/blob/master/doc/cnn_model.md)를 참고하였습니다. 문장을 [제품명, 브랜드, 모델명, 제조사명]으로 대입하여 적용하면 됩니다.

네트워크에서 변경된 사항으로, 각 Convolution은 bias를 초기화하지 않으며 활성화 함수 ReLU를 ELU로 변경, Batch normalization을 추가하였습니다.

### 계층적 추론전략

카카오 아레나 쇼핑 카테고리 분류 데이터셋의 정답은 대/중/소/세의 카테고리를 가지고 있습니다. catekitten에서는 이 4가지 계층적 카테고리에 대한 분류기를 별도로 학습하고 추론합니다. 이 때 대분류 이외의 카테고리에 대한 추론이 이루어질 경우, `catekitten.hierarchy` 에서 train.chunk.0x를 데이터를 기반으로 정의한 카테고리에 대한 부모 계층 정보(Ex: 세분류 카테고리에 대한 소/중/대분류 계층의 카테고리, 반드시 종속적이진 않음)를 활용하여 1번 데이터에 대해

대분류 예측 = 1  
* 학습 데이터중 세분류 카테고리 [4, 5, 6, 7, 8, 9, 10, 11, 12]번의 부모였음

중분류 예측 = 2  
* 학습 데이터중 세분류 카테고리 [4, 5, 6, 7, 8, 9]번의 부모였음

소분류 예측 = 3  
* 학습 데이터중 세분류 카테고리 [10, 11, 12]번의 부모였음

라는 카테고리 예측이 이루어졌을 경우 세분류 카테고리의 예측 결과의 softmax 행렬 중 [4, 5, 6, 7, 8, 9, 10, 11, 12]번의 값에는 가중치 b가 곱해지고, [4, 5, 6, 7, 8, 9]번의 값에는 가중치 m이 곱해지고, [10, 11, 12]번의 값에는 가중치 s가 곱해진 상태에서 argmax 연산을 통해 최종 label값을 도출합니다. 

### 계층간 전이학습

모든 학습데이터에 대해서, `bcateid`(대분류) 라벨과 `mcateid`(중분류) 라벨 데이터가 존재하지만 `scateid`(소분류) 라벨과 `dcateid`(세분류) 라벨의 데이터는 다수가 누락되어 있습니다. 모든 라벨에 대해서 주어진 데이터를 최대로 활용하기 위해, 계층간 전이학습(Transfer Learning) 전략을 활용합니다.

#### 학습 방법

1. 대분류 분류기 학습  
2. 대분류 분류기의 학습된 가중치를 이용해 중분류 분류기 학습  
3. 중분류 분류기의 학습된 가중치를 이용해 소분류 분류기 학습  
3. 소분소 분류기의 학습된 가중치를 이용해 세분류 분류기 학습


## Evaluation

((대분류 정확도) * 1.0 + (중분류 정확도) * 1.2 + (소분류 정확도) * 1.3 + (세분류 정확도) * 1.4) / 4

## References

[[Yoon Kim 2014](http://www.aclweb.org/anthology/D14-1181)] Convolutional Neural Networks for Sentence Classification

## Copyright

아래 모듈은 Best10님의 허가를 받고 [Kakao-Valhala 저장소](https://github.com/Demiguises/Kakao-Valhalla)의 코드를 차용/변경하였습니다.

```
catekitten.data  
catekitten.transform  
```

### License

Apache License 2.0

Copyright 2018-2019 (c) nyanye