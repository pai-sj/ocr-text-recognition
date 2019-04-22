CRNN, An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition
---

### 1. Objective

> 텐서플로우로 구현한 [CRNN](https://arxiv.org/abs/1507.05717) 모델입니다. 이 모델은 Text Recognition을 위해 만들어진 모델입니다. 

### 2. TODO-List

* [x] CRNN 골격 네트워크 구현 (CNN - RNN - transcription) 

* [x] Loss Function 구현

* [x] Data Generator 구현

* [x] MNIST 모델로 학습

* [ ] Text Recognition 데이터셋으로 학습

* [ ] Decode Layer 수정

### 3. 설명

현재 어떤 식으로 구현되어 있는 지는 `scripts/` 아래의 폴더를 참고하시면 됩니다. <br>
그리고 모델의 구성에 관련된 모든 코드는 `models/crnn.py`에 정리하였습니다. <br>
데이터셋은 aws S3에 저장되어 있습니다. 자동으로 다운받을 수 있도록 구현되어 있습니다.    

