"""
LSTM을 사용한 노이즈 예측 예제
"""

import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

from src.preprocessing.noise_generator import NoiseGenerator
from src.preprocessing.feature_extractor import FeatureExtractor
from src.models.lstm_model import LSTMNoisePredictor
from src.utils.visualization import plot_training_history


def create_sequences(data, time_steps=100):
    """
    시퀀스 데이터 생성

    Args:
        data: 입력 데이터 (n_samples, n_features)
        time_steps: 시퀀스 길이

    Returns:
        (X, y): 입력 시퀀스와 타겟
    """
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps])
    return np.array(X), np.array(y)


def main():
    print("=" * 60)
    print("LSTM 기반 노이즈 예측 예제")
    print("=" * 60)

    # 설정
    SAMPLE_RATE = 16000
    DURATION = 10.0  # 초
    TIME_STEPS = 100
    N_MELS = 128

    # 1. 노이즈 데이터 생성
    print("\n1. 노이즈 데이터 생성 중...")
    noise_gen = NoiseGenerator(sample_rate=SAMPLE_RATE)

    # 여러 종류의 노이즈 생성
    noises = []
    noise_types = ['white', 'pink', 'brown']

    for noise_type in noise_types:
        if noise_type == 'white':
            noise = noise_gen.generate_white_noise(DURATION)
        elif noise_type == 'pink':
            noise = noise_gen.generate_pink_noise(DURATION)
        elif noise_type == 'brown':
            noise = noise_gen.generate_brown_noise(DURATION)

        noises.append(noise)

    all_noise = np.concatenate(noises)
    print(f"   - 총 노이즈 샘플: {len(all_noise)} samples")

    # 2. 특징 추출
    print("\n2. Mel Spectrogram 특징 추출 중...")
    feature_extractor = FeatureExtractor(sample_rate=SAMPLE_RATE)

    mel_spec = feature_extractor.extract_mel_spectrogram(
        all_noise,
        n_mels=N_MELS
    )

    # 전치 (time, features)
    mel_spec = mel_spec.T
    print(f"   - 특징 shape: {mel_spec.shape}")

    # 3. 시퀀스 데이터 생성
    print("\n3. 시퀀스 데이터 생성 중...")
    X, y = create_sequences(mel_spec, time_steps=TIME_STEPS)
    print(f"   - X shape: {X.shape}")
    print(f"   - y shape: {y.shape}")

    # 학습/검증 분할
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"   - 학습 데이터: {X_train.shape[0]} samples")
    print(f"   - 검증 데이터: {X_val.shape[0]} samples")

    # 4. LSTM 모델 생성 및 학습
    print("\n4. LSTM 모델 학습 중...")

    model = LSTMNoisePredictor(
        input_shape=(TIME_STEPS, N_MELS),
        lstm_units=[128, 64],
        dropout_rate=0.2,
        learning_rate=0.001
    )

    print("\n모델 구조:")
    model.summary()

    print("\n학습 시작...")
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=20,  # 예제에서는 짧게
        batch_size=32
    )

    print("\n학습 완료!")

    # 5. 예측 테스트
    print("\n5. 예측 테스트...")
    test_sample = X_val[:5]
    predictions = model.predict(test_sample)

    print(f"   - 테스트 샘플 shape: {test_sample.shape}")
    print(f"   - 예측 결과 shape: {predictions.shape}")

    # 6. 모델 저장
    model_path = 'models/lstm_noise_predictor.h5'
    os.makedirs('models', exist_ok=True)
    model.save(model_path)
    print(f"\n6. 모델 저장: {model_path}")

    # 7. 학습 히스토리 시각화
    print("\n7. 학습 히스토리 시각화...")
    fig = plot_training_history(history, metrics=['loss', 'mae'])
    plt.savefig('lstm_training_history.png', dpi=150)
    print("   - 학습 히스토리 저장: lstm_training_history.png")

    plt.show()

    print("\n" + "=" * 60)
    print("예제 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
