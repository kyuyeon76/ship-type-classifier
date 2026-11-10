# 이미지 기반 머신러닝을 활용한 선박 종류 분류 (8조)

2026년 1학기 수업 팀 프로젝트 과제입니다.
선박 사진을 컨테이너선(container), 여객선(passenger), 유조선(tanker) 세 종류로 분류하는 모델을 scikit-learn으로 만들고, 모델 4개와 전처리 2가지를 비교했습니다.

코드는 `8조_학습코드 .ipynb` 노트북 하나에 모두 들어 있습니다.

## 데이터

- 구글 이미지 검색으로 팀원들이 직접 모은 사진 280장 (3개 클래스)
- 사진은 이 저장소에 포함하지 않았습니다.

## 전처리

1. 64×64로 크기 조정
2. 흑백(Grayscale) 또는 컬러(RGB) 변환
3. 1차원으로 펴기(Flatten) → 흑백 4,096차원 / 컬러 12,288차원
4. 0~1 범위로 정규화

## 실험

Random Forest, SVM, KNN, Logistic Regression 4개 모델을 흑백/컬러 데이터에 각각 학습시켰습니다.
학습:테스트 = 8:2 (테스트 56장), `random_state=123123`.

| 모델 | 흑백 | 컬러 |
|---|---|---|
| Random Forest | 53.57% | 53.57% |
| SVM (RBF, C=10) | 58.93% | **64.29%** |
| KNN | 30.36% | 51.79% |
| Logistic Regression | 55.36% | 55.36% |

컬러 + SVM 조합이 가장 높았습니다.

## 결과 (컬러 + SVM)

| 클래스 | precision | recall | f1-score | 테스트 수 |
|---|---|---|---|---|
| container | 0.60 | 0.65 | 0.62 | 23 |
| passenger | 0.80 | 0.75 | 0.77 | 16 |
| tanker | 0.56 | 0.53 | 0.55 | 17 |

여객선은 비교적 잘 구분했고, 유조선이 가장 많이 틀렸습니다. 노트북에 혼동 행렬과 테스트 이미지 8장의 예측 결과 그림이 있습니다.

## 실행 방법

Google Colab에서 실행하도록 작성했습니다.

1. 구글 드라이브에 아래 구조로 이미지를 넣습니다.
   ```
   MyDrive/Colab Notebooks/dataset/
   ├── container/
   ├── passenger/
   └── tanker/
   ```
2. 노트북을 Colab에서 열고 위에서부터 차례로 실행합니다.

사용 라이브러리: numpy, opencv-python, scikit-learn, matplotlib, seaborn
