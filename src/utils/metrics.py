"""
오디오 품질 평가 메트릭
"""

import numpy as np
from typing import Optional


def calculate_snr(
    clean_signal: np.ndarray,
    noisy_signal: np.ndarray,
    epsilon: float = 1e-10
) -> float:
    """
    Signal-to-Noise Ratio (SNR) 계산

    Args:
        clean_signal: 깨끗한 신호
        noisy_signal: 노이즈가 포함된 신호
        epsilon: 수치 안정성을 위한 작은 값

    Returns:
        SNR (dB)
    """
    noise = noisy_signal - clean_signal
    signal_power = np.mean(clean_signal ** 2)
    noise_power = np.mean(noise ** 2)

    snr = 10 * np.log10((signal_power + epsilon) / (noise_power + epsilon))
    return snr


def calculate_pesq(
    reference: np.ndarray,
    degraded: np.ndarray,
    sample_rate: int = 16000
) -> Optional[float]:
    """
    PESQ (Perceptual Evaluation of Speech Quality) 계산

    Note: 이 함수는 pesq 라이브러리가 필요합니다.
          pip install pesq

    Args:
        reference: 참조 신호
        degraded: 품질이 저하된 신호
        sample_rate: 샘플링 레이트

    Returns:
        PESQ 점수 (또는 None if error)
    """
    try:
        from pesq import pesq
        score = pesq(sample_rate, reference, degraded, 'wb')  # wideband
        return score
    except ImportError:
        print("Warning: pesq library not installed. Install with: pip install pesq")
        return None
    except Exception as e:
        print(f"Error calculating PESQ: {str(e)}")
        return None


def calculate_stoi(
    reference: np.ndarray,
    degraded: np.ndarray,
    sample_rate: int = 16000
) -> Optional[float]:
    """
    STOI (Short-Time Objective Intelligibility) 계산

    Note: 이 함수는 pystoi 라이브러리가 필요합니다.
          pip install pystoi

    Args:
        reference: 참조 신호
        degraded: 품질이 저하된 신호
        sample_rate: 샘플링 레이트

    Returns:
        STOI 점수 (또는 None if error)
    """
    try:
        from pystoi import stoi
        score = stoi(reference, degraded, sample_rate, extended=False)
        return score
    except ImportError:
        print("Warning: pystoi library not installed. Install with: pip install pystoi")
        return None
    except Exception as e:
        print(f"Error calculating STOI: {str(e)}")
        return None


def calculate_mse(signal1: np.ndarray, signal2: np.ndarray) -> float:
    """
    Mean Squared Error (MSE) 계산

    Args:
        signal1: 첫 번째 신호
        signal2: 두 번째 신호

    Returns:
        MSE
    """
    return np.mean((signal1 - signal2) ** 2)


def calculate_rmse(signal1: np.ndarray, signal2: np.ndarray) -> float:
    """
    Root Mean Squared Error (RMSE) 계산

    Args:
        signal1: 첫 번째 신호
        signal2: 두 번째 신호

    Returns:
        RMSE
    """
    return np.sqrt(calculate_mse(signal1, signal2))


def calculate_si_sdr(
    reference: np.ndarray,
    estimate: np.ndarray,
    epsilon: float = 1e-10
) -> float:
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) 계산

    Args:
        reference: 참조 신호
        estimate: 추정 신호
        epsilon: 수치 안정성을 위한 작은 값

    Returns:
        SI-SDR (dB)
    """
    # 스케일 인자 계산
    alpha = np.dot(estimate, reference) / (np.dot(reference, reference) + epsilon)

    # 스케일된 참조 신호
    scaled_reference = alpha * reference

    # 왜곡 계산
    distortion = estimate - scaled_reference

    # SI-SDR 계산
    si_sdr = 10 * np.log10(
        (np.sum(scaled_reference ** 2) + epsilon) /
        (np.sum(distortion ** 2) + epsilon)
    )

    return si_sdr


def calculate_spectral_distance(
    signal1: np.ndarray,
    signal2: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 2048
) -> float:
    """
    스펙트럴 거리 계산

    Args:
        signal1: 첫 번째 신호
        signal2: 두 번째 신호
        sample_rate: 샘플링 레이트
        n_fft: FFT 크기

    Returns:
        스펙트럴 거리
    """
    import librosa

    # STFT 계산
    stft1 = np.abs(librosa.stft(signal1, n_fft=n_fft))
    stft2 = np.abs(librosa.stft(signal2, n_fft=n_fft))

    # 로그 스펙트럼
    log_stft1 = np.log(stft1 + 1e-10)
    log_stft2 = np.log(stft2 + 1e-10)

    # 유클리드 거리
    distance = np.sqrt(np.mean((log_stft1 - log_stft2) ** 2))

    return distance


def evaluate_noise_reduction(
    clean: np.ndarray,
    noisy: np.ndarray,
    enhanced: np.ndarray,
    sample_rate: int = 16000
) -> dict:
    """
    노이즈 제거 성능 종합 평가

    Args:
        clean: 깨끗한 신호
        noisy: 노이즈가 포함된 신호
        enhanced: 노이즈 제거된 신호
        sample_rate: 샘플링 레이트

    Returns:
        평가 메트릭 딕셔너리
    """
    metrics = {}

    # SNR 개선
    snr_before = calculate_snr(clean, noisy)
    snr_after = calculate_snr(clean, enhanced)
    metrics['snr_improvement'] = snr_after - snr_before
    metrics['snr_before'] = snr_before
    metrics['snr_after'] = snr_after

    # SI-SDR
    metrics['si_sdr_before'] = calculate_si_sdr(clean, noisy)
    metrics['si_sdr_after'] = calculate_si_sdr(clean, enhanced)
    metrics['si_sdr_improvement'] = metrics['si_sdr_after'] - metrics['si_sdr_before']

    # MSE & RMSE
    metrics['mse_before'] = calculate_mse(clean, noisy)
    metrics['mse_after'] = calculate_mse(clean, enhanced)
    metrics['rmse_before'] = calculate_rmse(clean, noisy)
    metrics['rmse_after'] = calculate_rmse(clean, enhanced)

    # PESQ (optional)
    pesq_before = calculate_pesq(clean, noisy, sample_rate)
    pesq_after = calculate_pesq(clean, enhanced, sample_rate)
    if pesq_before is not None and pesq_after is not None:
        metrics['pesq_before'] = pesq_before
        metrics['pesq_after'] = pesq_after
        metrics['pesq_improvement'] = pesq_after - pesq_before

    # STOI (optional)
    stoi_before = calculate_stoi(clean, noisy, sample_rate)
    stoi_after = calculate_stoi(clean, enhanced, sample_rate)
    if stoi_before is not None and stoi_after is not None:
        metrics['stoi_before'] = stoi_before
        metrics['stoi_after'] = stoi_after
        metrics['stoi_improvement'] = stoi_after - stoi_before

    return metrics
