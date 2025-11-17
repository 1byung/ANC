"""
LSTM 기반 노이즈 예측 모델
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Optional


class LSTMNoisePredictor:
    """LSTM을 사용한 노이즈 패턴 예측 모델"""

    def __init__(
        self,
        input_shape: Tuple[int, int],
        lstm_units: list = [128, 64],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001
    ):
        """
        Args:
            input_shape: 입력 shape (time_steps, features)
            lstm_units: LSTM 레이어의 유닛 수 리스트
            dropout_rate: 드롭아웃 비율
            learning_rate: 학습률
        """
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self) -> keras.Model:
        """
        LSTM 모델 구축

        Returns:
            Keras Model
        """
        inputs = layers.Input(shape=self.input_shape)
        x = inputs

        # LSTM 레이어
        for i, units in enumerate(self.lstm_units):
            return_sequences = (i < len(self.lstm_units) - 1)
            x = layers.LSTM(
                units,
                return_sequences=return_sequences,
                dropout=self.dropout_rate
            )(x)
            x = layers.BatchNormalization()(x)

        # Dense 레이어
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        outputs = layers.Dense(self.input_shape[1], activation='linear')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # 컴파일
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )

        return model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        callbacks: Optional[list] = None
    ) -> keras.callbacks.History:
        """
        모델 학습

        Args:
            X_train: 학습 데이터
            y_train: 학습 레이블
            X_val: 검증 데이터
            y_val: 검증 레이블
            epochs: 에폭 수
            batch_size: 배치 크기
            callbacks: 콜백 리스트

        Returns:
            학습 히스토리
        """
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)

        if callbacks is None:
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss' if validation_data else 'loss',
                    patience=10,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss' if validation_data else 'loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7
                )
            ]

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        노이즈 예측

        Args:
            X: 입력 데이터

        Returns:
            예측된 노이즈
        """
        return self.model.predict(X)

    def save(self, filepath: str):
        """
        모델 저장

        Args:
            filepath: 저장 경로
        """
        self.model.save(filepath)

    def load(self, filepath: str):
        """
        모델 로드

        Args:
            filepath: 모델 경로
        """
        self.model = keras.models.load_model(filepath)

    def summary(self):
        """모델 요약 출력"""
        return self.model.summary()


class BiLSTMNoisePredictor(LSTMNoisePredictor):
    """Bidirectional LSTM을 사용한 노이즈 예측 모델"""

    def _build_model(self) -> keras.Model:
        """
        Bidirectional LSTM 모델 구축

        Returns:
            Keras Model
        """
        inputs = layers.Input(shape=self.input_shape)
        x = inputs

        # Bidirectional LSTM 레이어
        for i, units in enumerate(self.lstm_units):
            return_sequences = (i < len(self.lstm_units) - 1)
            x = layers.Bidirectional(
                layers.LSTM(
                    units,
                    return_sequences=return_sequences,
                    dropout=self.dropout_rate
                )
            )(x)
            x = layers.BatchNormalization()(x)

        # Dense 레이어
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        outputs = layers.Dense(self.input_shape[1], activation='linear')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # 컴파일
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )

        return model
