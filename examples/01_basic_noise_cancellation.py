"""
기본 노이즈 제거 예제
적응 필터를 사용한 능동 소음 제어
"""

import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

from src.preprocessing.noise_generator import NoiseGenerator
from src.models.adaptive_filter import ANCSystem
from src.utils.metrics import evaluate_noise_reduction
from src.utils.visualization import plot_comparison, plot_error_convergence


def main():
    print("=" * 60)
    print("능동 소음 제어 (Active Noise Control) 기본 예제")
    print("=" * 60)

    # 설정
    SAMPLE_RATE = 16000
    DURATION = 2.0  # 초
    SNR_DB = 5.0

    # 1. 노이즈 생성기 초기화
    print("\n1. 노이즈 생성 중...")
    noise_gen = NoiseGenerator(sample_rate=SAMPLE_RATE)

    # 깨끗한 신호 생성 (예: 사인파 조합)
    clean_signal = (
        noise_gen.generate_sine_wave(440, DURATION, amplitude=0.3) +  # A4
        noise_gen.generate_sine_wave(554, DURATION, amplitude=0.2)    # C#5
    )

    # 노이즈 생성
    noise = noise_gen.generate_pink_noise(DURATION, amplitude=0.1)

    # 노이즈 추가
    noisy_signal, noise_added = noise_gen.add_noise(clean_signal, noise, SNR_DB)

    print(f"   - 깨끗한 신호 생성: {len(clean_signal)} samples")
    print(f"   - 노이즈 추가 (SNR: {SNR_DB} dB)")

    # 2. 적응 필터를 사용한 노이즈 제거
    print("\n2. 적응 필터 (NLMS) 적용 중...")

    # ANC 시스템 초기화
    anc = ANCSystem(
        filter_type='nlms',
        filter_length=256,
        mu=0.5,
        epsilon=1e-6
    )

    # 노이즈 제거
    enhanced_signal, estimated_noise = anc.cancel_noise(noisy_signal, noise)

    print(f"   - 노이즈 제거 완료")

    # 3. 성능 평가
    print("\n3. 성능 평가:")
    metrics = evaluate_noise_reduction(
        clean_signal,
        noisy_signal,
        enhanced_signal,
        sample_rate=SAMPLE_RATE
    )

    print(f"   SNR 개선: {metrics['snr_improvement']:.2f} dB")
    print(f"   - 이전: {metrics['snr_before']:.2f} dB")
    print(f"   - 이후: {metrics['snr_after']:.2f} dB")
    print(f"\n   SI-SDR 개선: {metrics['si_sdr_improvement']:.2f} dB")
    print(f"   - 이전: {metrics['si_sdr_before']:.2f} dB")
    print(f"   - 이후: {metrics['si_sdr_after']:.2f} dB")

    # 4. 시각화
    print("\n4. 결과 시각화 중...")

    # 비교 플롯
    fig1 = plot_comparison(
        clean_signal,
        noisy_signal,
        enhanced_signal,
        sample_rate=SAMPLE_RATE
    )
    plt.savefig('noise_cancellation_comparison.png', dpi=150)
    print("   - 비교 플롯 저장: noise_cancellation_comparison.png")

    # 에러 수렴 플롯
    error = noisy_signal - estimated_noise
    fig2 = plot_error_convergence(error, sample_rate=SAMPLE_RATE)
    plt.savefig('error_convergence.png', dpi=150)
    print("   - 에러 수렴 플롯 저장: error_convergence.png")

    plt.show()

    print("\n" + "=" * 60)
    print("예제 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
