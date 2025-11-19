# Noise Control AI - 코드 구조 및 실행 가이드

## 📋 목차
1. [프로젝트 전체 구조](#프로젝트-전체-구조)
2. [실행 순서](#실행-순서)
3. [핵심 모듈 상세](#핵심-모듈-상세)
4. [실행 스크립트](#실행-스크립트)
5. [예제 코드](#예제-코드)
6. [데이터 흐름](#데이터-흐름)

---

## 프로젝트 전체 구조

```
noise-control-ai/
│
├── src/                    # 핵심 라이브러리 (재사용 가능한 모듈)
│   ├── preprocessing/      # 데이터 전처리
│   │   ├── audio_loader.py          # 오디오 파일 입출력
│   │   ├── feature_extractor.py     # 특징 추출 (STFT, Mel, MFCC)
│   │   └── noise_generator.py       # 노이즈 생성
│   │
│   ├── models/            # 모델 정의
│   │   ├── adaptive_filter.py       # LMS, NLMS, RLS 적응 필터
│   │   └── lstm_model.py            # LSTM 노이즈 예측 모델
│   │
│   └── utils/             # 유틸리티
│       ├── metrics.py                # 평가 메트릭 (SNR, SI-SDR, PESQ)
│       └── visualization.py          # 시각화 함수
│
├── scripts/               # 실행 스크립트 (독립 실행)
│   ├── download_datasets.py         # 데이터셋 다운로드
│   └── analyze_dataset.py           # 데이터셋 분석
│
├── examples/              # 예제 코드 (학습용)
│   ├── 01_basic_noise_cancellation.py   # 적응 필터 예제
│   ├── 02_lstm_noise_prediction.py      # LSTM 학습 예제
│   └── 03_pyspark_data_processing.py    # PySpark 병렬 처리
│
├── configs/               # 설정 파일
│   └── default.yaml
│
└── data/                  # 데이터 디렉토리
    ├── datasets/          # 원본 데이터셋
    ├── raw/              # 원본 오디오
    └── processed/        # 전처리된 데이터
```

---

## 실행 순서

### 일반적인 워크플로우

```
┌─────────────────────────────────────────────────────┐
│ 1단계: 데이터 수집                                    │
│ python scripts/download_datasets.py --dataset all   │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ 2단계: 데이터 분석                                    │
│ python scripts/analyze_dataset.py --dataset <path>  │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ 3단계: 예제 실행                                      │
│ - 01: 적응 필터로 노이즈 제거                          │
│ - 02: LSTM 모델 학습                                 │
│ - 03: PySpark 대규모 처리                             │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ 4단계: 실제 모델 학습 (향후)                          │
│ src/ 모듈들을 import해서 사용                         │
└─────────────────────────────────────────────────────┘
```

---

## 핵심 모듈 상세

### 1. NoiseGenerator (src/preprocessing/noise_generator.py)

**역할**: 테스트용 노이즈 신호 생성

**클래스 구조**:
```python
NoiseGenerator(sample_rate=16000)
    │
    ├─ generate_white_noise(duration, amplitude)
    │   └─ np.random.normal() 사용
    │
    ├─ generate_pink_noise(duration, amplitude)
    │   └─ Voss-McCartney 알고리즘 (여러 주파수 대역 합성)
    │
    ├─ generate_brown_noise(duration, amplitude)
    │   └─ 백색 노이즈의 누적합 (적분)
    │
    ├─ generate_sine_wave(frequency, duration, amplitude, phase)
    │   └─ 순수 사인파 생성
    │
    ├─ generate_band_limited_noise(duration, low_freq, high_freq, amplitude)
    │   └─ FFT로 특정 주파수 대역만 필터링
    │
    ├─ add_noise(clean_signal, noise, snr_db)
    │   └─ SNR(dB)에 맞춰 노이즈 크기 자동 조절
    │
    └─ generate_impulse_noise(duration, probability, amplitude)
        └─ 랜덤 임펄스 노이즈
```

**핵심 알고리즘**:

1. **White Noise**: 정규분포 랜덤 샘플
   ```python
   noise = np.random.normal(0, amplitude, num_samples)
   ```

2. **Pink Noise (1/f)**: Voss-McCartney 알고리즘
   ```python
   # 16개의 다른 주파수 대역을 합성
   for i in range(16):
       step = 2 ** i
       array[i] = np.repeat(np.random.randn(n // step + 1), step)
   pink = np.sum(array, axis=0)
   ```

3. **SNR 기반 노이즈 추가**:
   ```python
   signal_rms = np.sqrt(np.mean(clean_signal**2))
   noise_rms = np.sqrt(np.mean(noise**2))
   snr_linear = 10 ** (snr_db / 20)
   noise_scaled = noise * (signal_rms / (noise_rms * snr_linear))
   noisy_signal = clean_signal + noise_scaled
   ```

---

### 2. AudioLoader (src/preprocessing/audio_loader.py)

**역할**: 오디오 파일 입출력

**클래스 구조**:
```python
AudioLoader(sample_rate=16000, mono=True)
    │
    ├─ load(file_path) → (audio, sr)
    │   └─ librosa.load() 사용 (자동 리샘플링)
    │
    ├─ load_segment(file_path, start, duration)
    │   └─ 파일의 일부만 로드
    │
    ├─ save(audio, file_path, sample_rate)
    │   └─ soundfile.write() 사용
    │
    ├─ normalize(audio, target_db)
    │   └─ RMS 기반 정규화
    │
    └─ get_duration(file_path)
        └─ librosa.get_duration()
```

**RMS 기반 정규화**:
```python
rms = np.sqrt(np.mean(audio**2))
current_db = 20 * np.log10(rms)
gain = 10 ** ((target_db - current_db) / 20)
normalized = audio * gain
```

---

### 3. FeatureExtractor (src/preprocessing/feature_extractor.py)

**역할**: 오디오 특징 추출

**클래스 구조**:
```python
FeatureExtractor(sample_rate=16000)
    │
    ├─ extract_stft(audio, n_fft, hop_length)
    │   └─ (magnitude, phase) 반환
    │
    ├─ extract_mel_spectrogram(audio, n_fft, hop_length, n_mels)
    │   └─ Mel 스케일 스펙트로그램
    │
    ├─ extract_mfcc(audio, n_mfcc, n_fft, hop_length)
    │   └─ Mel-Frequency Cepstral Coefficients
    │
    ├─ extract_spectral_features(audio)
    │   ├─ spectral_centroid
    │   ├─ spectral_rolloff
    │   ├─ spectral_bandwidth
    │   ├─ zero_crossing_rate
    │   └─ rms
    │
    ├─ extract_chroma(audio, n_chroma)
    │   └─ 음정 특징
    │
    └─ istft(magnitude, phase, hop_length)
        └─ STFT 역변환 (신호 복원)
```

**STFT (Short-Time Fourier Transform)**:
- 시간-주파수 도메인 변환
- 윈도우를 이동시키며 FFT 수행

**Mel Spectrogram**:
- 사람 귀의 주파수 인식을 모방
- 저주파수는 세밀하게, 고주파수는 넓게

---

### 4. AdaptiveFilter (src/models/adaptive_filter.py) ⭐ 핵심

**역할**: 능동 소음 제어의 핵심 알고리즘

**클래스 계층**:
```python
AdaptiveFilter (기본 클래스)
    ├─ weights: np.array  # 필터 가중치
    └─ reset()           # 가중치 초기화
         │
         ├─ LMSFilter (Least Mean Squares)
         │   ├─ mu: 학습률
         │   └─ filter(reference, desired) → (output, error, weights_history)
         │
         ├─ NLMSFilter (Normalized LMS)
         │   ├─ mu: 학습률
         │   ├─ epsilon: 수치 안정성
         │   └─ filter(reference, desired) → (output, error, weights_history)
         │
         └─ RLSFilter (Recursive Least Squares)
             ├─ lambda_factor: 망각 인자
             ├─ P: 상관 행렬
             └─ filter(reference, desired) → (output, error, weights_history)

ANCSystem (통합 시스템)
    ├─ filter: LMSFilter | NLMSFilter | RLSFilter
    └─ cancel_noise(noisy_signal, reference_noise) → (clean_signal, estimated_noise)
```

**LMS 알고리즘 동작 원리**:

```python
# 매 샘플마다 반복:
for i in range(n_samples):
    # 1. 입력 버퍼 업데이트 (FIFO)
    buffer = np.roll(buffer, 1)
    buffer[0] = reference[i]

    # 2. 필터 출력 (노이즈 추정)
    output[i] = np.dot(weights, buffer)

    # 3. 에러 계산
    error[i] = desired[i] - output[i]

    # 4. 가중치 업데이트 (경사 하강법)
    weights += mu * error[i] * buffer  # ⭐ 핵심
```

**신호 흐름**:
```
참조 노이즈 ────┐
                ↓
            [적응 필터]
                ↓
            추정된 노이즈
                ↓
노이즈 신호 ──(減)────> 깨끗한 신호 (에러)
                ↑
            피드백으로 필터 학습
```

**NLMS vs LMS**:
- NLMS: 학습률을 입력 신호 크기로 정규화
  ```python
  norm = np.dot(buffer, buffer) + epsilon
  mu_normalized = mu / norm
  weights += mu_normalized * error[i] * buffer
  ```
- 더 안정적이고 빠른 수렴

**RLS (가장 빠른 수렴)**:
- Kalman gain 사용
- 계산 복잡도 높음 (O(N²))

---

### 5. LSTMNoisePredictor (src/models/lstm_model.py)

**역할**: LSTM으로 노이즈 패턴 학습 및 예측

**모델 구조**:
```python
LSTMNoisePredictor(
    input_shape=(time_steps, features),  # 예: (100, 128)
    lstm_units=[128, 64],
    dropout_rate=0.2,
    learning_rate=0.001
)

# Keras 모델:
Input(shape=(100, 128))
    ↓
LSTM(128, return_sequences=True, dropout=0.2)
    ↓
BatchNormalization()
    ↓
LSTM(64, return_sequences=False, dropout=0.2)
    ↓
BatchNormalization()
    ↓
Dense(64, activation='relu')
    ↓
Dropout(0.2)
    ↓
Dense(128, activation='linear')  # 출력: 다음 프레임 예측
```

**학습 방법**:
```python
model.compile(
    optimizer='adam',
    loss='mse',  # Mean Squared Error
    metrics=['mae']
)

history = model.fit(
    X_train,  # (samples, 100, 128)
    y_train,  # (samples, 128)
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[EarlyStopping, ReduceLROnPlateau]
)
```

**BiLSTM 버전**:
- 양방향 LSTM (과거+미래 정보 사용)
- 오프라인 처리에 적합

---

### 6. Metrics (src/utils/metrics.py)

**평가 지표들**:

```python
# 1. SNR (Signal-to-Noise Ratio)
snr_db = 10 * log10(signal_power / noise_power)

# 2. SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)
alpha = dot(estimate, reference) / dot(reference, reference)
scaled_reference = alpha * reference
distortion = estimate - scaled_reference
si_sdr = 10 * log10(sum(scaled_reference²) / sum(distortion²))

# 3. MSE, RMSE
mse = mean((signal1 - signal2)²)
rmse = sqrt(mse)

# 4. PESQ (Perceptual Evaluation of Speech Quality)
# - 음성 품질 평가 (사람 귀 기준)
# - 범위: -0.5 ~ 4.5

# 5. STOI (Short-Time Objective Intelligibility)
# - 음성 명료도 평가
# - 범위: 0 ~ 1

# 종합 평가
evaluate_noise_reduction(clean, noisy, enhanced)
    → {
        'snr_improvement': float,
        'si_sdr_improvement': float,
        'pesq_before': float,
        'pesq_after': float,
        ...
    }
```

---

## 실행 스크립트

### scripts/download_datasets.py

**실행 순서**:
```python
DatasetDownloader(data_dir='data/datasets')
    │
    ├─ 1. download_file(url, filename)
    │   └─ requests.get() + tqdm 진행바
    │
    ├─ 2. extract_archive(filepath)
    │   └─ zipfile.ZipFile() 또는 tarfile.open()
    │
    ├─ 3. download_esc50()
    │   └─ ESC-50-master.zip → 2,000 files
    │
    ├─ 4. download_librispeech_test_clean()
    │   └─ test-clean.tar.gz → 음성 데이터
    │
    ├─ 5. download_sample_noises()  # ⭐ src 모듈 사용
    │   └─ NoiseGenerator로 합성 노이즈 생성
    │       ├─ white_noise.wav
    │       ├─ pink_noise.wav
    │       ├─ brown_noise.wav
    │       └─ band_limited_noise.wav
    │
    └─ 6. show_info()
        └─ 다운로드된 데이터셋 통계 출력
```

**사용법**:
```bash
# 모든 데이터셋
python scripts/download_datasets.py --dataset all

# 특정 데이터셋만
python scripts/download_datasets.py --dataset esc50
python scripts/download_datasets.py --dataset samples
python scripts/download_datasets.py --dataset librispeech
```

---

### scripts/analyze_dataset.py

**실행 순서**:
```python
DatasetAnalyzer(dataset_path, sample_rate=16000)
    │
    ├─ 1. find_audio_files(['.wav', '.flac', '.mp3'])
    │   └─ Path.rglob() 사용
    │
    ├─ 2. analyze_files(max_files=None)
    │   └─ for each file:
    │       ├─ AudioLoader.load(filepath)  # ⭐ src 모듈
    │       └─ 통계 계산:
    │           ├─ duration = len(audio) / sr
    │           ├─ rms = sqrt(mean(audio²))
    │           ├─ peak = max(|audio|)
    │           └─ zero_crossings = 신호가 0 교차하는 횟수
    │   → Pandas DataFrame 반환
    │
    ├─ 3. print_statistics(df)
    │   └─ 평균, 중간값, 최소, 최대 출력
    │
    ├─ 4. visualize_distribution(df)
    │   └─ 6개 히스토그램:
    │       ├─ Duration
    │       ├─ RMS
    │       ├─ Peak
    │       ├─ Sample Rate
    │       ├─ File Size
    │       └─ Zero Crossings
    │
    └─ 5. visualize_samples(num_samples=5)
        └─ 랜덤 샘플 선택 → 파형 + 스펙트로그램
```

**사용법**:
```bash
# ESC-50 전체 분석
python scripts/analyze_dataset.py --dataset data/datasets/ESC-50-master/audio

# 처음 100개만 분석
python scripts/analyze_dataset.py \
    --dataset data/datasets/ESC-50-master/audio \
    --max-files 100

# 샘플 노이즈 분석
python scripts/analyze_dataset.py \
    --dataset data/datasets/sample_noises \
    --visualize-samples 4
```

**출력 파일**:
- `dataset_analysis.csv`: 통계 데이터
- `dataset_analysis.png`: 6개 히스토그램
- `dataset_samples/`: 샘플 파형 및 스펙트로그램

---

## 예제 코드

### examples/01_basic_noise_cancellation.py

**완전한 노이즈 제거 데모**

**실행 순서**:
```python
main():
    # 1단계: 테스트 신호 생성
    noise_gen = NoiseGenerator(sample_rate=16000)

    clean_signal = (
        noise_gen.generate_sine_wave(440, 2.0, 0.3) +   # A4 음
        noise_gen.generate_sine_wave(554, 2.0, 0.2)     # C#5 음
    )

    noise = noise_gen.generate_pink_noise(2.0, 0.1)

    noisy_signal, _ = noise_gen.add_noise(
        clean_signal,
        noise,
        snr_db=5.0
    )

    # 2단계: 적응 필터로 노이즈 제거 ⭐
    anc = ANCSystem(
        filter_type='nlms',
        filter_length=256,
        mu=0.5,
        epsilon=1e-6
    )

    enhanced_signal, estimated_noise = anc.cancel_noise(
        noisy_signal,
        noise  # 참조 노이즈
    )

    # 내부 동작:
    # NLMSFilter.filter(reference=noise, desired=noisy_signal)
    #   for each sample:
    #       output = weights · buffer
    #       error = desired - output
    #       norm = buffer · buffer + epsilon
    #       weights += (mu/norm) × error × buffer
    #   return enhanced_signal = error

    # 3단계: 성능 평가
    metrics = evaluate_noise_reduction(
        clean_signal,
        noisy_signal,
        enhanced_signal
    )

    print(f"SNR 개선: {metrics['snr_improvement']:.2f} dB")
    print(f"SI-SDR 개선: {metrics['si_sdr_improvement']:.2f} dB")

    # 4단계: 시각화
    plot_comparison(clean_signal, noisy_signal, enhanced_signal)
    plot_error_convergence(error)
```

**사용하는 모듈**:
- ✅ NoiseGenerator (src/preprocessing)
- ✅ ANCSystem, NLMSFilter (src/models)
- ✅ evaluate_noise_reduction (src/utils/metrics)
- ✅ plot_comparison (src/utils/visualization)

**실행**:
```bash
python examples/01_basic_noise_cancellation.py
```

**출력**:
- `noise_cancellation_comparison.png`: 3개 신호 비교
- `error_convergence.png`: 필터 학습 곡선

---

### examples/02_lstm_noise_prediction.py

**LSTM으로 노이즈 패턴 학습**

**실행 순서**:
```python
main():
    # 1단계: 노이즈 데이터 생성 (30초)
    noises = []
    for noise_type in ['white', 'pink', 'brown']:
        noise = noise_gen.generate_XXX_noise(10.0)
        noises.append(noise)
    all_noise = np.concatenate(noises)

    # 2단계: Mel Spectrogram 특징 추출
    feature_extractor = FeatureExtractor(sample_rate=16000)
    mel_spec = feature_extractor.extract_mel_spectrogram(
        all_noise,
        n_mels=128
    )
    # shape: (n_mels, time) → transpose → (time, n_mels)
    mel_spec = mel_spec.T  # (time, 128)

    # 3단계: 시퀀스 데이터 생성
    def create_sequences(data, time_steps=100):
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data[i:i+time_steps])  # 100 프레임
            y.append(data[i+time_steps])    # 다음 1 프레임
        return np.array(X), np.array(y)

    X, y = create_sequences(mel_spec, time_steps=100)
    # X shape: (samples, 100, 128)
    # y shape: (samples, 128)

    # 4단계: 학습/검증 분할
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # 5단계: LSTM 모델 생성 및 학습 ⭐
    model = LSTMNoisePredictor(
        input_shape=(100, 128),
        lstm_units=[128, 64],
        dropout_rate=0.2,
        learning_rate=0.001
    )

    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=20,
        batch_size=32
    )

    # 6단계: 모델 저장
    model.save('models/lstm_noise_predictor.h5')

    # 7단계: 학습 곡선 시각화
    plot_training_history(history, metrics=['loss', 'mae'])
```

**실행**:
```bash
python examples/02_lstm_noise_prediction.py
```

**출력**:
- `models/lstm_noise_predictor.h5`: 학습된 모델
- `lstm_training_history.png`: 학습 곡선

---

### examples/03_pyspark_data_processing.py

**대규모 데이터 병렬 처리**

**실행 순서**:
```python
main():
    # 1단계: Spark 세션 생성
    spark = SparkSession.builder \
        .appName("NoiseControlAI") \
        .master("local[*]") \
        .getOrCreate()

    # 2단계: 샘플 데이터 생성 (100개)
    data = []
    for i in range(100):
        if i % 3 == 0:
            noise = noise_gen.generate_white_noise(1.0)
        elif i % 3 == 1:
            noise = noise_gen.generate_pink_noise(1.0)
        else:
            noise = noise_gen.generate_brown_noise(1.0)

        data.append({
            'id': i,
            'noise_type': noise_type,
            'audio': noise.tolist()  # numpy → list
        })

    # 3단계: Pandas → Spark DataFrame
    df_pandas = pd.DataFrame(data)
    df_spark = spark.createDataFrame(df_pandas)

    # 4단계: UDF 정의 (User Defined Function)
    @udf(ArrayType(FloatType()))
    def extract_features(audio_list):
        audio = np.array(audio_list)
        feature_extractor = FeatureExtractor(sample_rate=16000)
        mel_spec = feature_extractor.extract_mel_spectrogram(
            audio,
            n_mels=40
        )
        return mel_spec.mean(axis=1).tolist()  # (40,)

    # 5단계: 병렬 특징 추출 ⭐
    df_with_features = df_spark.withColumn(
        'mel_features',
        extract_features(df_spark['audio'])
    )
    # Spark가 자동으로 병렬 처리!

    # 6단계: Parquet 저장 (압축 컬럼 형식)
    df_with_features.write \
        .mode('overwrite') \
        .parquet('data/processed/pyspark_features')

    # 7단계: 저장된 데이터 로드
    df_loaded = spark.read.parquet('data/processed/pyspark_features')
    print(f"로드된 레코드 수: {df_loaded.count()}")

    spark.stop()
```

**왜 PySpark?**
- 수천~수만 개 파일을 병렬로 처리
- 메모리 효율적 (디스크 기반)
- Parquet: 컬럼 기반 압축 형식 (빠른 읽기)

**실행**:
```bash
python examples/03_pyspark_data_processing.py
```

---

## 데이터 흐름

### 워크플로우 1: 데이터 수집 및 분석

```
download_datasets.py
    ↓
data/datasets/
    ├─ ESC-50-master/audio/  (2,000 .wav files)
    └─ sample_noises/        (4 .wav files)
    ↓
analyze_dataset.py
    ↓
    ├─ dataset_analysis.csv
    ├─ dataset_analysis.png
    └─ dataset_samples/
```

### 워크플로우 2: 적응 필터 노이즈 제거

```
01_basic_noise_cancellation.py
    ↓
NoiseGenerator
    ├─ clean_signal (사인파)
    ├─ noise (핑크 노이즈)
    └─ noisy_signal = clean + noise
    ↓
ANCSystem (NLMS Filter)
    └─ for each sample:
        1. output = weights · buffer
        2. error = desired - output
        3. weights += (μ/norm) × error × buffer
    ↓
enhanced_signal (깨끗해진 신호)
    ↓
evaluate_noise_reduction()
    └─ SNR, SI-SDR 개선도 측정
```

### 워크플로우 3: LSTM 학습

```
02_lstm_noise_prediction.py
    ↓
NoiseGenerator → 노이즈 생성 (30초)
    ↓
FeatureExtractor → Mel Spectrogram (time, 128)
    ↓
create_sequences() → (samples, 100, 128)
    ↓
LSTMNoisePredictor
    └─ Input(100, 128)
       → LSTM(128) → LSTM(64)
       → Dense(128)
    ↓
model.fit(X_train, y_train)
    ↓
models/lstm_noise_predictor.h5
```

### 워크플로우 4: PySpark 병렬 처리

```
03_pyspark_data_processing.py
    ↓
NoiseGenerator → 100개 노이즈 생성
    ↓
Pandas DataFrame
    ↓
Spark DataFrame
    ↓
UDF(extract_features)
    └─ FeatureExtractor.extract_mel_spectrogram()
    ↓
병렬 처리 (자동)
    ↓
Parquet 저장
    └─ data/processed/pyspark_features/
```

---

## 모듈 의존성 맵

```
scripts/download_datasets.py
    └─ import NoiseGenerator (src/preprocessing)

scripts/analyze_dataset.py
    ├─ import AudioLoader (src/preprocessing)
    └─ import plot_waveform, plot_spectrogram (src/utils)

examples/01_basic_noise_cancellation.py
    ├─ import NoiseGenerator (src/preprocessing)
    ├─ import ANCSystem, NLMSFilter (src/models)
    ├─ import evaluate_noise_reduction (src/utils/metrics)
    └─ import plot_comparison, plot_error_convergence (src/utils)

examples/02_lstm_noise_prediction.py
    ├─ import NoiseGenerator (src/preprocessing)
    ├─ import FeatureExtractor (src/preprocessing)
    ├─ import LSTMNoisePredictor (src/models)
    └─ import plot_training_history (src/utils)

examples/03_pyspark_data_processing.py
    ├─ import NoiseGenerator (src/preprocessing)
    └─ import FeatureExtractor (src/preprocessing)
```

---

## 핵심 개념 정리

### 1. 능동 소음 제어 (ANC) 원리

```
                    참조 마이크
                        │
                        ↓ (참조 노이즈 측정)
                  [적응 필터]
                        │
                        ↓ (상쇄 신호 생성)
실제 노이즈 ──→  (減) ──→ 깨끗한 신호
                  ↑
                  │
              에러 마이크
              (피드백으로 필터 학습)
```

### 2. 적응 필터 학습 과정

```
초기 상태: weights = [0, 0, 0, ..., 0]

반복 (매 샘플):
    1. 노이즈 추정 = weights · buffer
    2. 에러 = 실제 - 추정
    3. weights 업데이트 (경사 하강)

시간이 지날수록:
    weights → 최적값
    에러 → 0
    노이즈 제거 성능 ↑
```

### 3. SNR (Signal-to-Noise Ratio)

```
SNR (dB) = 10 × log₁₀(신호 전력 / 노이즈 전력)

예시:
- SNR = 0 dB   → 신호와 노이즈 크기 동일
- SNR = 10 dB  → 신호가 노이즈보다 10배 강함
- SNR = -5 dB  → 노이즈가 신호보다 강함 (나쁨)
```

### 4. STFT (Short-Time Fourier Transform)

```
시간 도메인:  [샘플1, 샘플2, 샘플3, ...]
              ↓ (윈도우를 이동하며 FFT)
시간-주파수:  [
               [주파수1, 주파수2, ...],  # 시간 0
               [주파수1, 주파수2, ...],  # 시간 1
               ...
              ]
```

---

## 자주 사용하는 명령어

### 데이터 다운로드
```bash
python scripts/download_datasets.py --dataset all
python scripts/download_datasets.py --dataset esc50
python scripts/download_datasets.py --dataset samples
```

### 데이터 분석
```bash
python scripts/analyze_dataset.py --dataset data/datasets/ESC-50-master/audio --max-files 100
python scripts/analyze_dataset.py --dataset data/datasets/sample_noises --visualize-samples 4
```

### 예제 실행
```bash
python examples/01_basic_noise_cancellation.py
python examples/02_lstm_noise_prediction.py
python examples/03_pyspark_data_processing.py
```

### 패키지 설치
```bash
pip install -r requirements.txt
```

---

## 문제 해결

### 1. librosa 설치 오류
```bash
# Windows
pip install librosa --upgrade

# 오디오 백엔드 설치
pip install soundfile
```

### 2. PySpark 실행 오류
```bash
# Java 설치 확인
java -version

# PySpark 재설치
pip uninstall pyspark
pip install pyspark
```

### 3. 메모리 부족
```python
# analyze_dataset.py
python scripts/analyze_dataset.py --dataset <path> --max-files 100

# PySpark
# configs/default.yaml에서 메모리 조절
pyspark:
  executor_memory: 2g  # 4g → 2g로 감소
  driver_memory: 1g
```

---

## 추가 학습 자료

### 적응 필터
- LMS Algorithm: https://en.wikipedia.org/wiki/Least_mean_squares_filter
- Adaptive Filtering Primer: [Widrow & Stearns, 1985]

### 오디오 처리
- librosa 튜토리얼: https://librosa.org/doc/main/tutorial.html
- 디지털 신호 처리: [Oppenheim & Schafer]

### 딥러닝
- LSTM Networks: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
- Keras 가이드: https://keras.io/guides/

---

## 라이선스

MIT License

---

**작성일**: 2025-11-19
**버전**: 1.0
