# Noise Control AI

능동 소음 제어(Active Noise Control, ANC) 시스템을 딥러닝과 신호 처리 기술을 활용하여 구현한 프로젝트입니다.

## 프로젝트 개요

이 프로젝트는 TensorFlow, NumPy, Pandas, PySpark를 활용하여 실시간 노이즈 제어 시스템을 구축합니다.

### 주요 기능

- **딥러닝 기반 노이즈 예측**: TensorFlow를 활용한 RNN/LSTM 모델로 노이즈 패턴 학습
- **신호 처리**: NumPy와 SciPy를 활용한 적응 필터링(Adaptive Filtering)
- **대규모 데이터 처리**: PySpark를 활용한 오디오 데이터 전처리
- **실시간 추론**: 저지연 노이즈 상쇄 신호 생성

## 기술 스택

- **Deep Learning**: TensorFlow, Keras
- **Data Processing**: NumPy, Pandas, PySpark
- **Audio Processing**: librosa, soundfile
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Development**: Jupyter Notebook, Python 3.10+

## 프로젝트 구조

```
noise-control-ai/
├── data/
│   ├── raw/              # 원본 오디오 데이터
│   └── processed/        # 전처리된 데이터
├── models/               # 학습된 모델 저장
├── notebooks/            # Jupyter 노트북 (실험 및 분석)
├── src/
│   ├── preprocessing/    # 데이터 전처리 모듈
│   ├── models/           # 모델 정의
│   ├── training/         # 학습 코드
│   ├── inference/        # 추론 코드
│   └── utils/            # 유틸리티 함수
├── tests/                # 테스트 코드
├── configs/              # 설정 파일
├── requirements.txt      # 의존성 패키지
└── README.md
```

## 설치 방법

### 1. 저장소 클론

```bash
git clone https://github.com/[your-username]/noise-control-ai.git
cd noise-control-ai
```

### 2. 가상 환경 생성 (권장)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

## 사용 방법

### 데이터 준비

```bash
python src/preprocessing/prepare_data.py --input data/raw --output data/processed
```

### 모델 학습

```bash
python src/training/train.py --config configs/default.yaml
```

### 추론 (노이즈 제거)

```bash
python src/inference/predict.py --input audio.wav --output clean_audio.wav
```

## 능동 소음 제어 원리

1. **적응 필터링**: LMS(Least Mean Squares) 알고리즘을 사용하여 노이즈 패턴 학습
2. **딥러닝 예측**: LSTM 네트워크로 미래 노이즈 신호 예측
3. **상쇄 신호 생성**: 예측된 노이즈의 역위상 신호 생성
4. **실시간 적용**: 저지연으로 상쇄 신호를 원본 신호에 합성

## 개발 로드맵

- [ ] 기본 데이터 파이프라인 구축
- [ ] 적응 필터 알고리즘 구현 (LMS, NLMS)
- [ ] LSTM 기반 노이즈 예측 모델 개발
- [ ] PySpark를 활용한 대규모 데이터 전처리
- [ ] 실시간 추론 시스템 구현
- [ ] 성능 평가 및 최적화
- [ ] Web/Mobile 데모 애플리케이션

## 기여 방법

Pull Request를 환영합니다!

## 라이선스

MIT License

## 참고 자료

- [Active Noise Control - Wikipedia](https://en.wikipedia.org/wiki/Active_noise_control)
- [TensorFlow Audio Tutorial](https://www.tensorflow.org/tutorials/audio)
- [Librosa Documentation](https://librosa.org/)
