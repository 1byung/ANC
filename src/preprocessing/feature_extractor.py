"""
오디오 특징 추출 모듈
"""

import numpy as np
import librosa
from typing import Tuple, Optional


class FeatureExtractor:
    """오디오에서 다양한 특징을 추출하는 클래스"""

    def __init__(self, sample_rate: int = 16000):
        """
        Args:
            sample_rate: 샘플링 레이트 (Hz)
        """
        self.sample_rate = sample_rate

    def extract_stft(
        self,
        audio: np.ndarray,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        STFT (Short-Time Fourier Transform) 추출

        Args:
            audio: 오디오 데이터
            n_fft: FFT 윈도우 크기
            hop_length: 홉 길이
            win_length: 윈도우 길이

        Returns:
            (magnitude, phase): 크기와 위상
        """
        stft = librosa.stft(
            audio,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length
        )
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        return magnitude, phase

    def extract_mel_spectrogram(
        self,
        audio: np.ndarray,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128
    ) -> np.ndarray:
        """
        Mel Spectrogram 추출

        Args:
            audio: 오디오 데이터
            n_fft: FFT 윈도우 크기
            hop_length: 홉 길이
            n_mels: Mel 필터 수

        Returns:
            Mel spectrogram
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        # dB 스케일로 변환
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db

    def extract_mfcc(
        self,
        audio: np.ndarray,
        n_mfcc: int = 40,
        n_fft: int = 2048,
        hop_length: int = 512
    ) -> np.ndarray:
        """
        MFCC (Mel-Frequency Cepstral Coefficients) 추출

        Args:
            audio: 오디오 데이터
            n_mfcc: MFCC 계수 수
            n_fft: FFT 윈도우 크기
            hop_length: 홉 길이

        Returns:
            MFCC features
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length
        )
        return mfcc

    def extract_spectral_features(self, audio: np.ndarray) -> dict:
        """
        다양한 스펙트럴 특징 추출

        Args:
            audio: 오디오 데이터

        Returns:
            특징 딕셔너리
        """
        features = {}

        # Spectral Centroid
        features['spectral_centroid'] = librosa.feature.spectral_centroid(
            y=audio,
            sr=self.sample_rate
        )[0]

        # Spectral Rolloff
        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
            y=audio,
            sr=self.sample_rate
        )[0]

        # Spectral Bandwidth
        features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(
            y=audio,
            sr=self.sample_rate
        )[0]

        # Zero Crossing Rate
        features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(audio)[0]

        # RMS Energy
        features['rms'] = librosa.feature.rms(y=audio)[0]

        return features

    def extract_chroma(self, audio: np.ndarray, n_chroma: int = 12) -> np.ndarray:
        """
        Chroma 특징 추출

        Args:
            audio: 오디오 데이터
            n_chroma: Chroma bin 수

        Returns:
            Chroma features
        """
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=self.sample_rate,
            n_chroma=n_chroma
        )
        return chroma

    @staticmethod
    def istft(
        magnitude: np.ndarray,
        phase: np.ndarray,
        hop_length: int = 512,
        win_length: Optional[int] = None
    ) -> np.ndarray:
        """
        STFT를 역변환하여 오디오 신호로 복원

        Args:
            magnitude: 크기
            phase: 위상
            hop_length: 홉 길이
            win_length: 윈도우 길이

        Returns:
            복원된 오디오
        """
        stft = magnitude * np.exp(1j * phase)
        audio = librosa.istft(
            stft,
            hop_length=hop_length,
            win_length=win_length
        )
        return audio
