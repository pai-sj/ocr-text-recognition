Text Recognition Model (CRNN + Seq2Seq + Attention + Jamo Embedding)
---

## 1. Objective

> 텐서플로우로 구현한  

## 2. installation

### 1. 가상환경 설치
```bash
virtualenv venv
source venv/bin/activate 
``` 
### 2. 필수 라이브러리 설치

GPU 환경 혹은 CPU 환경에 따라 텐서플로우를 설치해 주세요.
````bash
pip install -r requirements.txt
pip install tensorflow==1.14 # or tensorflow-gpu
````

## 3. Usage

### 1. 디렉토리 구조

현재 어떤 식으로 구현되어 있는 지는 `scripts/` 아래의 폴더를 참고하시면 됩니다. <br>
````markdown
models/
   |- generator.py : Keras의 `Data Generator` 클래스가 구현된 스크립트 
   |- jamo.py : 한글 자모자를 다루는 메소드들이 구현된 스크립트
   |- layers.py : Text Recognition Model에 관련된 custom Layer들이 구현된 스크립트
   |- losses.py : Text Recognition Model에 관련된 Custom Losses들이 구현된 스크립트 
   |- optimizer.py : Custom Optimizer, ADAM이 구현된 스크립트
   |- utils.py : 기타 시각화 혹은 generator에 이용되는 스크립트
````
