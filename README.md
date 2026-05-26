# Watermelon Classifier (간단 가이드)

이 프로젝트는 수박 이미지로 생장 정도(미숙, 적당함, 과숙)를 분류하는 예제입니다.

폴더 구조(권장):

- dataset/train/미숙/  <- 미숙 클래스 이미지들을 넣으세요
- dataset/train/적당함/ <- 적당함 클래스 이미지들을 넣으세요
- dataset/train/과숙/ <- 과숙 클래스 이미지들을 넣으세요
- dataset/test/ <- 테스트용 이미지
- saved/ <- 학습된 모델 저장

레이블 목록 파일: `classes.txt` (한 줄에 하나의 라벨)

설치 (Windows, PowerShell):

```powershell
python -m pip install --upgrade pip
pip install opencv-python torch torchvision tqdm
```

간단 사용법:

1. `dataset/train/` 아래에 각 클래스별 폴더에 이미지를 넣습니다.
2. `train.py`를 실행하여 모델을 학습합니다 (추후 제공).
3. `predict.py <이미지경로>`로 단일 이미지 추론을 수행합니다 (추후 제공).

팁:
- 각 클래스당 가능한 다양한 이미지(각도, 조명)을 넣으세요.
- 데이터가 적으면 `transforms.RandomHorizontalFlip`, `ColorJitter` 같은 증강을 사용하세요.

