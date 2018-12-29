# Catekitten

Catekitten은 Kakao-arena(1. 쇼핑몰 카테고리 분류) 참가에 사용된 모든 코드와 아이디어를 공개하는 저장소입니다.

## 모델 재구현
```
# 저장소 clone
git clone https://github.com/nyanye/catekitten.git
cd catekitten

# HDF preprocessing
python -m catekitten.data merge "./data/raw/" "./data/prep/textonly.h5" 

# Train
python -m catekitten.train
```

## Evaluation

((대분류 정확도) * 1.0 + (중분류 정확도) * 1.2 + (소분류 정확도) * 1.3 + (세분류 정확도) * 1.4) / 4

## Copyright

아래 모듈은 [Kakao-Valhala 저장소](https://github.com/Demiguises/Kakao-Valhalla)의 코드를 차용/변경하였습니다.

```
catekitten.data  
catekitten.transform  
```

### License

Apache License 2.0

Copyright 2018-2019 (c) nyanye