# Shopping categories

## Data

hdf5로 chunk 데이터 입력.  
y = ['bcateid', 'mcateid', 'scateid', 'dcateid']  
x = ['brand', 'img_feat', 'maker',  'model', 'pid', 'price', 'product',  'updttm']  

### Feature

Text feature: 브랜드명, 제조사, 정제된 상품명, 상품 ID, 상품명, 상품 정보
Linear scale: 업데이트 시간, 가격
Image feature: 이미지 피쳐(2048)

### Label

대분류, 중분류, 소분류, 세분류

## Preprocessing

업데이트 시간 = unix timestamp로 치환, linear scale feature로 사용

## Evaluation

((대분류 정확도) * 1.0 + (중분류 정확도) * 1.2 + (소분류 정확도) * 1.3 + (세분류 정확도) * 1.4) / 4
