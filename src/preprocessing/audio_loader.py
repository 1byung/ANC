"""
오디오 파일 로더 모듈
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Optional
from pathlib import Path


class AudioLoader:
    """오디오 파일을 로드하고 기본 전처리를 수행하는 클래스"""

    def __init__(self, sample_rate: int = 16000, mono: bool = True):
        """
        Args:
            sample_rate: 샘플링 레이트 (Hz)
            mono: 모노로 변환 여부
        """
        self.sample_rate = sample_rate
        self.mono = mono

    def load(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        오디오 파일을 로드합니다.

        Args:
            file_path: 오디오 파일 경로

        Returns:
            (audio_data, sample_rate): 오디오 데이터와 샘플링 레이트
        """
        try:
            audio, sr = librosa.load(
                file_path,
                sr=self.sample_rate,
                mono=self.mono
            )
            return audio, sr
        except Exception as e:
            raise ValueError(f"오디오 파일 로드 실패: {file_path}\n{str(e)}")

    def load_segment(
        self,
        file_path: str,
        start: float = 0.0,
        duration: Optional[float] = None
    ) -> Tuple[np.ndarray, int]:
        """
        오디오 파일의 일부분을 로드합니다.

        Args:
            file_path: 오디오 파일 경로
            start: 시작 시간 (초)
            duration: 지속 시간 (초), None이면 끝까지

        Returns:
            (audio_data, sample_rate): 오디오 데이터와 샘플링 레이트
        """
        try:
            audio, sr = librosa.load(
                file_path,
                sr=self.sample_rate,
                mono=self.mono,
                offset=start,
                duration=duration
            )
            return audio, sr
        except Exception as e:
            raise ValueError(f"오디오 세그먼트 로드 실패: {file_path}\n{str(e)}")

    def save(self, audio: np.ndarray, file_path: str, sample_rate: Optional[int] = None):
        """
        오디오 데이터를 파일로 저장합니다.

        Args:
            audio: 오디오 데이터
            file_path: 저장할 파일 경로
            sample_rate: 샘플링 레이트 (None이면 self.sample_rate 사용)
        """
        sr = sample_rate or self.sample_rate
        try:
            sf.write(file_path, audio, sr)
        except Exception as e:
            raise ValueError(f"오디오 파일 저장 실패: {file_path}\n{str(e)}")

    @staticmethod
    def normalize(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
        """
        오디오를 정규화합니다.

        Args:
            audio: 오디오 데이터
            target_db: 목표 데시벨

        Returns:
            정규화된 오디오 데이터
        """
        # RMS 기반 정규화
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            current_db = 20 * np.log10(rms)
            gain = 10 ** ((target_db - current_db) / 20)
            return audio * gain
        return audio

    @staticmethod
    def get_duration(file_path: str) -> float:
        """
        오디오 파일의 길이를 반환합니다.

        Args:
            file_path: 오디오 파일 경로

        Returns:
            오디오 길이 (초)
        """
        return librosa.get_duration(path=file_path)
