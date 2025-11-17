"""
시각화 유틸리티
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from typing import Optional, Tuple


def plot_waveform(
    audio: np.ndarray,
    sample_rate: int = 16000,
    title: str = "Waveform",
    figsize: Tuple[int, int] = (12, 4)
) -> plt.Figure:
    """
    오디오 파형 플롯

    Args:
        audio: 오디오 데이터
        sample_rate: 샘플링 레이트
        title: 그래프 제목
        figsize: 그림 크기

    Returns:
        Figure 객체
    """
    fig, ax = plt.subplots(figsize=figsize)

    librosa.display.waveshow(audio, sr=sample_rate, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_spectrogram(
    audio: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 2048,
    hop_length: int = 512,
    title: str = "Spectrogram",
    figsize: Tuple[int, int] = (12, 6),
    cmap: str = 'viridis'
) -> plt.Figure:
    """
    스펙트로그램 플롯

    Args:
        audio: 오디오 데이터
        sample_rate: 샘플링 레이트
        n_fft: FFT 크기
        hop_length: 홉 길이
        title: 그래프 제목
        figsize: 그림 크기
        cmap: 컬러맵

    Returns:
        Figure 객체
    """
    fig, ax = plt.subplots(figsize=figsize)

    # STFT 계산
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # 스펙트로그램 표시
    img = librosa.display.specshow(
        S_db,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis='time',
        y_axis='hz',
        ax=ax,
        cmap=cmap
    )

    ax.set_title(title)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')

    plt.tight_layout()
    return fig


def plot_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    title: str = "Mel Spectrogram",
    figsize: Tuple[int, int] = (12, 6),
    cmap: str = 'viridis'
) -> plt.Figure:
    """
    Mel 스펙트로그램 플롯

    Args:
        audio: 오디오 데이터
        sample_rate: 샘플링 레이트
        n_fft: FFT 크기
        hop_length: 홉 길이
        n_mels: Mel 필터 수
        title: 그래프 제목
        figsize: 그림 크기
        cmap: 컬러맵

    Returns:
        Figure 객체
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Mel 스펙트로그램 계산
    S = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    # 표시
    img = librosa.display.specshow(
        S_db,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis='time',
        y_axis='mel',
        ax=ax,
        cmap=cmap
    )

    ax.set_title(title)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')

    plt.tight_layout()
    return fig


def plot_comparison(
    clean: np.ndarray,
    noisy: np.ndarray,
    enhanced: np.ndarray,
    sample_rate: int = 16000,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    깨끗한/노이즈/향상된 신호 비교 플롯

    Args:
        clean: 깨끗한 신호
        noisy: 노이즈가 포함된 신호
        enhanced: 노이즈 제거된 신호
        sample_rate: 샘플링 레이트
        figsize: 그림 크기

    Returns:
        Figure 객체
    """
    fig, axes = plt.subplots(3, 2, figsize=figsize)

    signals = [
        (clean, "Clean Signal"),
        (noisy, "Noisy Signal"),
        (enhanced, "Enhanced Signal")
    ]

    for i, (signal, title) in enumerate(signals):
        # 파형
        librosa.display.waveshow(signal, sr=sample_rate, ax=axes[i, 0])
        axes[i, 0].set_title(f"{title} - Waveform")
        axes[i, 0].set_xlabel("Time (s)")
        axes[i, 0].set_ylabel("Amplitude")
        axes[i, 0].grid(True, alpha=0.3)

        # 스펙트로그램
        D = librosa.stft(signal)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        img = librosa.display.specshow(
            S_db,
            sr=sample_rate,
            x_axis='time',
            y_axis='hz',
            ax=axes[i, 1],
            cmap='viridis'
        )
        axes[i, 1].set_title(f"{title} - Spectrogram")
        fig.colorbar(img, ax=axes[i, 1], format='%+2.0f dB')

    plt.tight_layout()
    return fig


def plot_training_history(
    history,
    metrics: list = ['loss', 'mae'],
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    학습 히스토리 플롯

    Args:
        history: Keras History 객체
        metrics: 플롯할 메트릭 리스트
        figsize: 그림 크기

    Returns:
        Figure 객체
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        if metric in history.history:
            axes[i].plot(history.history[metric], label=f'Training {metric}')

        val_metric = f'val_{metric}'
        if val_metric in history.history:
            axes[i].plot(history.history[val_metric], label=f'Validation {metric}')

        axes[i].set_title(f'{metric.upper()} over epochs')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel(metric.upper())
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_filter_response(
    weights: np.ndarray,
    sample_rate: int = 16000,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    적응 필터의 주파수 응답 플롯

    Args:
        weights: 필터 가중치
        sample_rate: 샘플링 레이트
        figsize: 그림 크기

    Returns:
        Figure 객체
    """
    from scipy import signal

    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # 임펄스 응답
    axes[0].stem(weights, basefmt=' ')
    axes[0].set_title('Filter Impulse Response')
    axes[0].set_xlabel('Sample')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)

    # 주파수 응답
    w, h = signal.freqz(weights, worN=8000, fs=sample_rate)
    axes[1].plot(w, 20 * np.log10(np.abs(h)))
    axes[1].set_title('Filter Frequency Response')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude (dB)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_error_convergence(
    error: np.ndarray,
    sample_rate: int = 16000,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    적응 필터 에러 수렴 플롯

    Args:
        error: 에러 신호
        sample_rate: 샘플링 레이트
        figsize: 그림 크기

    Returns:
        Figure 객체
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 에러 신호
    time = np.arange(len(error)) / sample_rate
    axes[0].plot(time, error)
    axes[0].set_title('Error Signal')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Error')
    axes[0].grid(True, alpha=0.3)

    # 에러 제곱 (학습 곡선)
    error_squared = error ** 2
    axes[1].plot(time, error_squared)
    axes[1].set_title('Squared Error (Learning Curve)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Squared Error')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
