"""
적응 필터 알고리즘 (LMS, NLMS, RLS)
"""

import numpy as np
from typing import Tuple, Optional


class AdaptiveFilter:
    """적응 필터 기본 클래스"""

    def __init__(self, filter_length: int):
        """
        Args:
            filter_length: 필터 길이
        """
        self.filter_length = filter_length
        self.weights = np.zeros(filter_length)

    def reset(self):
        """필터 가중치 초기화"""
        self.weights = np.zeros(self.filter_length)


class LMSFilter(AdaptiveFilter):
    """LMS (Least Mean Squares) 적응 필터"""

    def __init__(self, filter_length: int, mu: float = 0.01):
        """
        Args:
            filter_length: 필터 길이
            mu: 학습률 (step size)
        """
        super().__init__(filter_length)
        self.mu = mu

    def filter(
        self,
        reference: np.ndarray,
        desired: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        LMS 필터링 수행

        Args:
            reference: 참조 신호 (노이즈)
            desired: 목표 신호 (노이즈가 포함된 신호)

        Returns:
            (output, error, weights_history): 출력, 에러, 가중치 히스토리
        """
        n_samples = len(reference)
        output = np.zeros(n_samples)
        error = np.zeros(n_samples)
        weights_history = np.zeros((n_samples, self.filter_length))

        # 입력 버퍼
        buffer = np.zeros(self.filter_length)

        for i in range(n_samples):
            # 버퍼 업데이트
            buffer = np.roll(buffer, 1)
            buffer[0] = reference[i]

            # 필터 출력
            output[i] = np.dot(self.weights, buffer)

            # 에러 계산
            error[i] = desired[i] - output[i]

            # 가중치 업데이트 (LMS 알고리즘)
            self.weights += self.mu * error[i] * buffer

            # 가중치 히스토리 저장
            weights_history[i] = self.weights.copy()

        return output, error, weights_history


class NLMSFilter(AdaptiveFilter):
    """NLMS (Normalized LMS) 적응 필터"""

    def __init__(self, filter_length: int, mu: float = 0.5, epsilon: float = 1e-6):
        """
        Args:
            filter_length: 필터 길이
            mu: 학습률 (0 < mu < 2)
            epsilon: 수치 안정성을 위한 작은 값
        """
        super().__init__(filter_length)
        self.mu = mu
        self.epsilon = epsilon

    def filter(
        self,
        reference: np.ndarray,
        desired: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        NLMS 필터링 수행

        Args:
            reference: 참조 신호 (노이즈)
            desired: 목표 신호 (노이즈가 포함된 신호)

        Returns:
            (output, error, weights_history): 출력, 에러, 가중치 히스토리
        """
        n_samples = len(reference)
        output = np.zeros(n_samples)
        error = np.zeros(n_samples)
        weights_history = np.zeros((n_samples, self.filter_length))

        # 입력 버퍼
        buffer = np.zeros(self.filter_length)

        for i in range(n_samples):
            # 버퍼 업데이트
            buffer = np.roll(buffer, 1)
            buffer[0] = reference[i]

            # 필터 출력
            output[i] = np.dot(self.weights, buffer)

            # 에러 계산
            error[i] = desired[i] - output[i]

            # 정규화된 학습률
            norm = np.dot(buffer, buffer) + self.epsilon
            mu_normalized = self.mu / norm

            # 가중치 업데이트 (NLMS 알고리즘)
            self.weights += mu_normalized * error[i] * buffer

            # 가중치 히스토리 저장
            weights_history[i] = self.weights.copy()

        return output, error, weights_history


class RLSFilter(AdaptiveFilter):
    """RLS (Recursive Least Squares) 적응 필터"""

    def __init__(
        self,
        filter_length: int,
        lambda_factor: float = 0.99,
        delta: float = 1.0
    ):
        """
        Args:
            filter_length: 필터 길이
            lambda_factor: 망각 인자 (0 < lambda <= 1)
            delta: 초기 P 행렬 스케일
        """
        super().__init__(filter_length)
        self.lambda_factor = lambda_factor
        self.P = np.eye(filter_length) * delta

    def filter(
        self,
        reference: np.ndarray,
        desired: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        RLS 필터링 수행

        Args:
            reference: 참조 신호 (노이즈)
            desired: 목표 신호 (노이즈가 포함된 신호)

        Returns:
            (output, error, weights_history): 출력, 에러, 가중치 히스토리
        """
        n_samples = len(reference)
        output = np.zeros(n_samples)
        error = np.zeros(n_samples)
        weights_history = np.zeros((n_samples, self.filter_length))

        # 입력 버퍼
        buffer = np.zeros(self.filter_length)

        for i in range(n_samples):
            # 버퍼 업데이트
            buffer = np.roll(buffer, 1)
            buffer[0] = reference[i]

            # 필터 출력
            output[i] = np.dot(self.weights, buffer)

            # 에러 계산
            error[i] = desired[i] - output[i]

            # Kalman gain 계산
            P_u = np.dot(self.P, buffer)
            k = P_u / (self.lambda_factor + np.dot(buffer, P_u))

            # 가중치 업데이트
            self.weights += k * error[i]

            # P 행렬 업데이트
            self.P = (self.P - np.outer(k, P_u)) / self.lambda_factor

            # 가중치 히스토리 저장
            weights_history[i] = self.weights.copy()

        return output, error, weights_history

    def reset(self):
        """필터 가중치 및 P 행렬 초기화"""
        super().reset()
        self.P = np.eye(self.filter_length)


class ANCSystem:
    """능동 소음 제어 시스템"""

    def __init__(
        self,
        filter_type: str = 'nlms',
        filter_length: int = 256,
        **filter_params
    ):
        """
        Args:
            filter_type: 필터 타입 ('lms', 'nlms', 'rls')
            filter_length: 필터 길이
            **filter_params: 필터별 추가 파라미터
        """
        self.filter_type = filter_type.lower()
        self.filter_length = filter_length

        # 필터 초기화
        if self.filter_type == 'lms':
            self.filter = LMSFilter(filter_length, **filter_params)
        elif self.filter_type == 'nlms':
            self.filter = NLMSFilter(filter_length, **filter_params)
        elif self.filter_type == 'rls':
            self.filter = RLSFilter(filter_length, **filter_params)
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")

    def cancel_noise(
        self,
        noisy_signal: np.ndarray,
        reference_noise: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        노이즈 제거 수행

        Args:
            noisy_signal: 노이즈가 포함된 신호
            reference_noise: 참조 노이즈 신호

        Returns:
            (clean_signal, estimated_noise): 깨끗한 신호와 추정된 노이즈
        """
        estimated_noise, error, _ = self.filter.filter(reference_noise, noisy_signal)

        # 에러 신호가 깨끗한 신호의 추정치
        clean_signal = error

        return clean_signal, estimated_noise

    def reset(self):
        """필터 리셋"""
        self.filter.reset()
