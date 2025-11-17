"""
데이터셋 분석 및 검증 스크립트
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import argparse

from src.preprocessing.audio_loader import AudioLoader
from src.utils.visualization import plot_waveform, plot_spectrogram


class DatasetAnalyzer:
    """데이터셋 분석기"""

    def __init__(self, dataset_path, sample_rate=16000):
        """
        Args:
            dataset_path: 데이터셋 경로
            sample_rate: 샘플링 레이트
        """
        self.dataset_path = Path(dataset_path)
        self.sample_rate = sample_rate
        self.audio_loader = AudioLoader(sample_rate=sample_rate)

    def find_audio_files(self, extensions=['.wav', '.flac', '.mp3']):
        """
        오디오 파일 찾기

        Args:
            extensions: 오디오 파일 확장자 리스트

        Returns:
            오디오 파일 경로 리스트
        """
        audio_files = []
        for ext in extensions:
            audio_files.extend(self.dataset_path.rglob(f'*{ext}'))
        return sorted(audio_files)

    def analyze_files(self, max_files=None):
        """
        오디오 파일 분석

        Args:
            max_files: 분석할 최대 파일 수

        Returns:
            분석 결과 DataFrame
        """
        audio_files = self.find_audio_files()

        if max_files:
            audio_files = audio_files[:max_files]

        print(f"\n분석할 파일 수: {len(audio_files)}")

        results = []

        for filepath in tqdm(audio_files, desc="파일 분석"):
            try:
                # 오디오 로드
                audio, sr = self.audio_loader.load(str(filepath))

                # 통계 계산
                duration = len(audio) / sr
                rms = np.sqrt(np.mean(audio ** 2))
                peak = np.max(np.abs(audio))
                zero_crossings = np.sum(np.abs(np.diff(np.sign(audio)))) / 2

                results.append({
                    'filename': filepath.name,
                    'path': str(filepath.relative_to(self.dataset_path)),
                    'duration': duration,
                    'samples': len(audio),
                    'sample_rate': sr,
                    'rms': rms,
                    'peak': peak,
                    'zero_crossings': zero_crossings,
                    'size_mb': filepath.stat().st_size / (1024 * 1024)
                })

            except Exception as e:
                print(f"\n[ERROR] 오류: {filepath.name} - {str(e)}")

        return pd.DataFrame(results)

    def print_statistics(self, df):
        """
        통계 정보 출력

        Args:
            df: 분석 결과 DataFrame
        """
        print("\n" + "=" * 60)
        print("데이터셋 통계")
        print("=" * 60)

        print(f"\n총 파일 수: {len(df)}")
        print(f"총 크기: {df['size_mb'].sum():.2f} MB")
        print(f"총 길이: {df['duration'].sum():.2f} 초 ({df['duration'].sum()/3600:.2f} 시간)")

        print("\n길이 통계:")
        print(f"  - 평균: {df['duration'].mean():.2f} 초")
        print(f"  - 중간값: {df['duration'].median():.2f} 초")
        print(f"  - 최소: {df['duration'].min():.2f} 초")
        print(f"  - 최대: {df['duration'].max():.2f} 초")

        print("\n샘플링 레이트:")
        sr_counts = df['sample_rate'].value_counts()
        for sr, count in sr_counts.items():
            print(f"  - {sr} Hz: {count}개 파일")

        print("\nRMS 통계:")
        print(f"  - 평균: {df['rms'].mean():.6f}")
        print(f"  - 중간값: {df['rms'].median():.6f}")
        print(f"  - 최소: {df['rms'].min():.6f}")
        print(f"  - 최대: {df['rms'].max():.6f}")

        print("\nPeak 통계:")
        print(f"  - 평균: {df['peak'].mean():.6f}")
        print(f"  - 중간값: {df['peak'].median():.6f}")
        print(f"  - 최소: {df['peak'].min():.6f}")
        print(f"  - 최대: {df['peak'].max():.6f}")

    def visualize_distribution(self, df, save_path='dataset_analysis.png'):
        """
        분포 시각화

        Args:
            df: 분석 결과 DataFrame
            save_path: 저장 경로
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # Duration distribution
        axes[0, 0].hist(df['duration'], bins=50, edgecolor='black')
        axes[0, 0].set_title('Duration Distribution')
        axes[0, 0].set_xlabel('Duration (seconds)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].grid(True, alpha=0.3)

        # RMS distribution
        axes[0, 1].hist(df['rms'], bins=50, edgecolor='black')
        axes[0, 1].set_title('RMS Distribution')
        axes[0, 1].set_xlabel('RMS')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].grid(True, alpha=0.3)

        # Peak distribution
        axes[0, 2].hist(df['peak'], bins=50, edgecolor='black')
        axes[0, 2].set_title('Peak Distribution')
        axes[0, 2].set_xlabel('Peak Amplitude')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].grid(True, alpha=0.3)

        # Sample rate distribution
        sr_counts = df['sample_rate'].value_counts()
        axes[1, 0].bar(range(len(sr_counts)), sr_counts.values)
        axes[1, 0].set_xticks(range(len(sr_counts)))
        axes[1, 0].set_xticklabels([f'{sr}Hz' for sr in sr_counts.index], rotation=45)
        axes[1, 0].set_title('Sample Rate Distribution')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # File size distribution
        axes[1, 1].hist(df['size_mb'], bins=50, edgecolor='black')
        axes[1, 1].set_title('File Size Distribution')
        axes[1, 1].set_xlabel('Size (MB)')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].grid(True, alpha=0.3)

        # Zero crossings
        axes[1, 2].hist(df['zero_crossings'], bins=50, edgecolor='black')
        axes[1, 2].set_title('Zero Crossings Distribution')
        axes[1, 2].set_xlabel('Zero Crossings')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"\n[OK] 시각화 저장: {save_path}")

    def visualize_samples(self, num_samples=5, save_dir='dataset_samples'):
        """
        샘플 오디오 시각화

        Args:
            num_samples: 시각화할 샘플 수
            save_dir: 저장 디렉토리
        """
        audio_files = self.find_audio_files()

        if len(audio_files) == 0:
            print("오디오 파일을 찾을 수 없습니다.")
            return

        # 랜덤 샘플 선택
        import random
        samples = random.sample(audio_files, min(num_samples, len(audio_files)))

        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)

        for i, filepath in enumerate(samples):
            try:
                audio, sr = self.audio_loader.load(str(filepath))

                # 파형 시각화
                fig1 = plot_waveform(audio, sr, title=f"Sample {i+1}: {filepath.name}")
                fig1.savefig(save_path / f'sample_{i+1}_waveform.png', dpi=150)
                plt.close(fig1)

                # 스펙트로그램 시각화
                fig2 = plot_spectrogram(audio, sr, title=f"Sample {i+1}: {filepath.name}")
                fig2.savefig(save_path / f'sample_{i+1}_spectrogram.png', dpi=150)
                plt.close(fig2)

                print(f"[OK] 샘플 {i+1} 시각화 완료: {filepath.name}")

            except Exception as e:
                print(f"[ERROR] 오류: {filepath.name} - {str(e)}")

        print(f"\n[OK] 샘플 시각화 저장: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='데이터셋 분석')
    parser.add_argument(
        '--dataset',
        required=True,
        help='데이터셋 경로'
    )
    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='분석할 최대 파일 수'
    )
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=16000,
        help='샘플링 레이트'
    )
    parser.add_argument(
        '--visualize-samples',
        type=int,
        default=5,
        help='시각화할 샘플 수'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("데이터셋 분석기")
    print("=" * 60)
    print(f"데이터셋: {args.dataset}")

    analyzer = DatasetAnalyzer(args.dataset, sample_rate=args.sample_rate)

    # 파일 분석
    df = analyzer.analyze_files(max_files=args.max_files)

    if len(df) == 0:
        print("\n[ERROR] 분석할 파일이 없습니다.")
        return

    # 통계 출력
    analyzer.print_statistics(df)

    # 분포 시각화
    analyzer.visualize_distribution(df)

    # 샘플 시각화
    analyzer.visualize_samples(num_samples=args.visualize_samples)

    # CSV로 저장
    csv_path = 'dataset_analysis.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] 분석 결과 저장: {csv_path}")

    print("\n" + "=" * 60)
    print("분석 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
