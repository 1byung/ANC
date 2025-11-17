"""
PySpark를 사용한 대규모 오디오 데이터 처리 예제
"""

import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, FloatType, StructType, StructField, StringType
import pandas as pd

from src.preprocessing.noise_generator import NoiseGenerator
from src.preprocessing.feature_extractor import FeatureExtractor


def main():
    print("=" * 60)
    print("PySpark를 사용한 대규모 오디오 데이터 처리")
    print("=" * 60)

    # 1. Spark 세션 생성
    print("\n1. Spark 세션 생성 중...")
    spark = SparkSession.builder \
        .appName("NoiseControlAI") \
        .master("local[*]") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

    print(f"   - Spark 버전: {spark.version}")

    # 2. 샘플 데이터 생성
    print("\n2. 샘플 오디오 데이터 생성 중...")

    SAMPLE_RATE = 16000
    DURATION = 1.0
    NUM_SAMPLES = 100  # 100개의 샘플 오디오 생성

    noise_gen = NoiseGenerator(sample_rate=SAMPLE_RATE)

    # Pandas DataFrame으로 샘플 데이터 생성
    data = []
    noise_types = ['white', 'pink', 'brown']

    for i in range(NUM_SAMPLES):
        noise_type = noise_types[i % len(noise_types)]

        if noise_type == 'white':
            noise = noise_gen.generate_white_noise(DURATION)
        elif noise_type == 'pink':
            noise = noise_gen.generate_pink_noise(DURATION)
        else:
            noise = noise_gen.generate_brown_noise(DURATION)

        data.append({
            'id': i,
            'noise_type': noise_type,
            'audio': noise.tolist()  # numpy array를 list로 변환
        })

    df_pandas = pd.DataFrame(data)
    print(f"   - 생성된 샘플 수: {len(df_pandas)}")

    # 3. Spark DataFrame으로 변환
    print("\n3. Spark DataFrame으로 변환 중...")
    df_spark = spark.createDataFrame(df_pandas)
    df_spark.printSchema()

    # 4. UDF 정의 - 특징 추출
    print("\n4. 특징 추출 UDF 정의...")

    def extract_features(audio_list):
        """Mel Spectrogram 특징 추출"""
        try:
            audio = np.array(audio_list, dtype=np.float32)
            feature_extractor = FeatureExtractor(sample_rate=SAMPLE_RATE)

            # Mel Spectrogram 추출
            mel_spec = feature_extractor.extract_mel_spectrogram(audio, n_mels=40)

            # 평균값 반환 (간단화)
            return mel_spec.mean(axis=1).tolist()
        except Exception as e:
            print(f"Error extracting features: {e}")
            return [0.0] * 40

    extract_features_udf = udf(extract_features, ArrayType(FloatType()))

    # 5. 병렬 특징 추출
    print("\n5. 병렬 특징 추출 수행 중...")

    df_with_features = df_spark.withColumn(
        'mel_features',
        extract_features_udf(df_spark['audio'])
    )

    # 6. 결과 확인
    print("\n6. 처리 결과:")
    df_result = df_with_features.select('id', 'noise_type', 'mel_features')
    df_result.show(10, truncate=False)

    # 7. 통계 정보
    print("\n7. 노이즈 타입별 통계:")
    df_stats = df_with_features.groupBy('noise_type').count()
    df_stats.show()

    # 8. 결과 저장 (Parquet 형식)
    output_path = 'data/processed/pyspark_features'
    print(f"\n8. 결과 저장 중: {output_path}")

    df_with_features.select('id', 'noise_type', 'mel_features').write \
        .mode('overwrite') \
        .parquet(output_path)

    print(f"   - 저장 완료: {output_path}")

    # 9. 저장된 데이터 로드 테스트
    print("\n9. 저장된 데이터 로드 테스트...")
    df_loaded = spark.read.parquet(output_path)
    print(f"   - 로드된 레코드 수: {df_loaded.count()}")

    # 10. Spark 세션 종료
    spark.stop()
    print("\n10. Spark 세션 종료")

    print("\n" + "=" * 60)
    print("예제 완료!")
    print("=" * 60)
    print("\n참고:")
    print("- PySpark는 대규모 데이터셋 처리에 유용합니다")
    print("- 수천~수만 개의 오디오 파일을 병렬로 처리할 수 있습니다")
    print("- Parquet 형식은 효율적인 저장과 빠른 로딩을 제공합니다")


if __name__ == "__main__":
    main()
