"""
노이즈 생성 모듈
"""

import numpy as np
from typing import Tuple, Optional


class NoiseGenerator:
    """다양한 종류의 노이즈를 생성하는 클래스"""

    def __init__(self, sample_rate: int = 16000):
        """
        Args:
            sample_rate: 샘플링 레이트 (Hz)
        """
        self.sample_rate = sample_rate

    def generate_white_noise(
        self,
        duration: float,
        amplitude: float = 0.1
    ) -> np.ndarray:
        """
        백색 노이즈(White Noise) 생성

        Args:
            duration: 지속 시간 (초)
            amplitude: 진폭

        Returns:
            노이즈 신호
        """
        num_samples = int(duration * self.sample_rate)
        noise = np.random.normal(0, amplitude, num_samples)
        return noise.astype(np.float32)

    def generate_pink_noise(
        self,
        duration: float,
        amplitude: float = 0.1
    ) -> np.ndarray:
        """
        핑크 노이즈(Pink Noise, 1/f noise) 생성

        Args:
            duration: 지속 시간 (초)
            amplitude: 진폭

        Returns:
            노이즈 신호
        """
        num_samples = int(duration * self.sample_rate)

        # Voss-McCartney 알고리즘
        num_rows = 16
        array = np.zeros((num_rows, num_samples))
        for i in range(num_rows):
            step = 2 ** i
            array[i] = np.repeat(
                np.random.randn(num_samples // step + 1),
                step
            )[:num_samples]

        pink = np.sum(array, axis=0)
        pink = pink / np.max(np.abs(pink)) * amplitude
        return pink.astype(np.float32)

    def generate_brown_noise(
        self,
        duration: float,
        amplitude: float = 0.1
    ) -> np.ndarray:
        """
        브라운 노이즈(Brown Noise, Red Noise) 생성

        Args:
            duration: 지속 시간 (초)
            amplitude: 진폭

        Returns:
            노이즈 신호
        """
        num_samples = int(duration * self.sample_rate)
        white_noise = np.random.randn(num_samples)
        brown = np.cumsum(white_noise)
        brown = brown / np.max(np.abs(brown)) * amplitude
        return brown.astype(np.float32)

    def generate_sine_wave(
        self,
        frequency: float,
        duration: float,
        amplitude: float = 0.1,
        phase: float = 0.0
    ) -> np.ndarray:
        """
        사인파 생성

        Args:
            frequency: 주파수 (Hz)
            duration: 지속 시간 (초)
            amplitude: 진폭
            phase: 초기 위상 (라디안)

        Returns:
            사인파 신호
        """
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        sine = amplitude * np.sin(2 * np.pi * frequency * t + phase)
        return sine.astype(np.float32)

    def generate_band_limited_noise(
        self,
        duration: float,
        low_freq: float,
        high_freq: float,
        amplitude: float = 0.1
    ) -> np.ndarray:
        """
        대역 제한 노이즈 생성

        Args:
            duration: 지속 시간 (초)
            low_freq: 하한 주파수 (Hz)
            high_freq: 상한 주파수 (Hz)
            amplitude: 진폭

        Returns:
            대역 제한 노이즈
        """
        num_samples = int(duration * self.sample_rate)

        # 백색 노이즈 생성
        white_noise = np.random.randn(num_samples)

        # FFT
        fft = np.fft.fft(white_noise)
        freq = np.fft.fftfreq(num_samples, 1/self.sample_rate)

        # 대역 필터링
        mask = (np.abs(freq) >= low_freq) & (np.abs(freq) <= high_freq)
        fft[~mask] = 0

        # IFFT
        filtered = np.fft.ifft(fft).real
        filtered = filtered / np.max(np.abs(filtered)) * amplitude
        return filtered.astype(np.float32)

    def add_noise(
        self,
        clean_signal: np.ndarray,
        noise: np.ndarray,
        snr_db: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        깨끗한 신호에 노이즈를 추가

        Args:
            clean_signal: 깨끗한 신호
            noise: 노이즈 신호
            snr_db: 신호 대 잡음비 (dB)

        Returns:
            (noisy_signal, noise): 노이즈가 추가된 신호와 사용된 노이즈
        """
        # 신호와 노이즈의 길이를 맞춤
        if len(noise) < len(clean_signal):
            # 노이즈를 반복
            repeats = int(np.ceil(len(clean_signal) / len(noise)))
            noise = np.tile(noise, repeats)[:len(clean_signal)]
        elif len(noise) > len(clean_signal):
            # 노이즈를 자름
            noise = noise[:len(clean_signal)]

        # RMS 계산
        signal_rms = np.sqrt(np.mean(clean_signal**2))
        noise_rms = np.sqrt(np.mean(noise**2))

        # SNR에 따른 노이즈 스케일 조정
        if noise_rms > 0:
            snr_linear = 10 ** (snr_db / 20)
            noise_scaled = noise * (signal_rms / (noise_rms * snr_linear))
        else:
            noise_scaled = noise

        # 신호에 노이즈 추가
        noisy_signal = clean_signal + noise_scaled

        return noisy_signal.astype(np.float32), noise_scaled.astype(np.float32)

    def generate_impulse_noise(
        self,
        duration: float,
        probability: float = 0.01,
        amplitude: float = 0.5
    ) -> np.ndarray:
        """
        임펄스 노이즈 생성

        Args:
            duration: 지속 시간 (초)
            probability: 임펄스 발생 확률
            amplitude: 임펄스 진폭

        Returns:
            임펄스 노이즈
        """
        num_samples = int(duration * self.sample_rate)
        noise = np.zeros(num_samples)

        # 랜덤하게 임펄스 생성
        impulse_indices = np.random.rand(num_samples) < probability
        noise[impulse_indices] = np.random.choice([-1, 1], size=np.sum(impulse_indices)) * amplitude

        return noise.astype(np.float32)
