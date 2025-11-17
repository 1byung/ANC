"""
공개 오디오 데이터셋 다운로드 스크립트

지원하는 데이터셋:
1. ESC-50: 환경 소리 (2,000 samples, 50 classes)
2. UrbanSound8K: 도시 환경 소음 (8,732 samples, 10 classes)
3. LibriSpeech: 깨끗한 음성 데이터
4. DNS Challenge: Microsoft 노이즈 제거 데이터셋
"""

import os
import sys
import requests
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm
import argparse


class DatasetDownloader:
    """오디오 데이터셋 다운로더"""

    def __init__(self, data_dir='data/datasets'):
        """
        Args:
            data_dir: 데이터를 저장할 디렉토리
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_file(self, url, filename, chunk_size=8192):
        """
        파일 다운로드 (진행 표시 포함)

        Args:
            url: 다운로드 URL
            filename: 저장할 파일명
            chunk_size: 청크 크기
        """
        filepath = self.data_dir / filename

        # 이미 다운로드된 경우 스킵
        if filepath.exists():
            print(f"[OK] 이미 다운로드됨: {filename}")
            return filepath

        print(f"다운로드 중: {filename}")

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(filepath, 'wb') as f:
                with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            print(f"[OK] 다운로드 완료: {filename}")
            return filepath

        except Exception as e:
            print(f"[ERROR] 다운로드 실패: {str(e)}")
            if filepath.exists():
                filepath.unlink()
            return None

    def extract_archive(self, filepath, extract_dir=None):
        """
        압축 파일 해제

        Args:
            filepath: 압축 파일 경로
            extract_dir: 압축 해제 디렉토리
        """
        if extract_dir is None:
            extract_dir = filepath.parent

        print(f"압축 해제 중: {filepath.name}")

        try:
            if filepath.suffix == '.zip':
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)

            elif filepath.suffix in ['.tar', '.gz', '.bz2', '.xz']:
                with tarfile.open(filepath, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_dir)

            print(f"[OK] 압축 해제 완료: {filepath.name}")
            return True

        except Exception as e:
            print(f"[ERROR] 압축 해제 실패: {str(e)}")
            return False

    def download_esc50(self):
        """
        ESC-50 데이터셋 다운로드
        - 환경 소리 50개 클래스
        - 2,000 samples (각 5초)
        - 44.1kHz, mono
        """
        print("\n" + "=" * 60)
        print("ESC-50 데이터셋 다운로드")
        print("=" * 60)

        url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
        filename = "ESC-50-master.zip"

        filepath = self.download_file(url, filename)

        if filepath:
            self.extract_archive(filepath)
            print("\n[OK] ESC-50 다운로드 및 압축 해제 완료")
            print(f"   경로: {self.data_dir / 'ESC-50-master'}")
            return True

        return False

    def download_urbansound8k(self):
        """
        UrbanSound8K 데이터셋 다운로드
        - 도시 환경 소음 10개 클래스
        - 8,732 samples

        Note: 수동 다운로드 필요 (라이선스 동의)
        """
        print("\n" + "=" * 60)
        print("UrbanSound8K 데이터셋")
        print("=" * 60)
        print("\n[WARNING] UrbanSound8K는 수동 다운로드가 필요합니다.")
        print("\n다운로드 방법:")
        print("1. https://urbansounddataset.weebly.com/urbansound8k.html 방문")
        print("2. 데이터셋 다운로드 및 동의")
        print("3. 다운로드한 파일을 다음 경로에 압축 해제:")
        print(f"   {self.data_dir}")

        return False

    def download_librispeech_test_clean(self):
        """
        LibriSpeech test-clean 데이터셋 다운로드
        - 깨끗한 영어 음성 데이터
        - 약 5시간 분량
        - 16kHz
        """
        print("\n" + "=" * 60)
        print("LibriSpeech test-clean 데이터셋 다운로드")
        print("=" * 60)

        url = "https://www.openslr.org/resources/12/test-clean.tar.gz"
        filename = "test-clean.tar.gz"

        filepath = self.download_file(url, filename)

        if filepath:
            self.extract_archive(filepath)
            print("\n[OK] LibriSpeech test-clean 다운로드 및 압축 해제 완료")
            print(f"   경로: {self.data_dir / 'LibriSpeech' / 'test-clean'}")
            return True

        return False

    def download_sample_noises(self):
        """
        샘플 노이즈 파일 생성 (합성)
        """
        print("\n" + "=" * 60)
        print("샘플 노이즈 생성")
        print("=" * 60)

        # src 디렉토리를 Python 경로에 추가
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

        try:
            from src.preprocessing.noise_generator import NoiseGenerator
            from src.preprocessing.audio_loader import AudioLoader

            sample_dir = self.data_dir / 'sample_noises'
            sample_dir.mkdir(exist_ok=True)

            noise_gen = NoiseGenerator(sample_rate=16000)
            audio_loader = AudioLoader(sample_rate=16000)

            duration = 10.0  # 10초

            noise_types = {
                'white_noise': noise_gen.generate_white_noise(duration, amplitude=0.1),
                'pink_noise': noise_gen.generate_pink_noise(duration, amplitude=0.1),
                'brown_noise': noise_gen.generate_brown_noise(duration, amplitude=0.1),
                'band_limited_noise': noise_gen.generate_band_limited_noise(
                    duration, 500, 2000, amplitude=0.1
                ),
            }

            for name, noise in noise_types.items():
                filepath = sample_dir / f"{name}.wav"
                audio_loader.save(noise, str(filepath))
                print(f"[OK] 생성: {name}.wav")

            print(f"\n[OK] 샘플 노이즈 생성 완료")
            print(f"   경로: {sample_dir}")
            return True

        except Exception as e:
            print(f"[ERROR] 샘플 노이즈 생성 실패: {str(e)}")
            return False

    def show_info(self):
        """다운로드된 데이터셋 정보 표시"""
        print("\n" + "=" * 60)
        print("다운로드된 데이터셋 정보")
        print("=" * 60)

        datasets = [
            'ESC-50-master',
            'UrbanSound8K',
            'LibriSpeech',
            'sample_noises'
        ]

        for dataset in datasets:
            dataset_path = self.data_dir / dataset
            if dataset_path.exists():
                # 파일 개수 세기
                audio_files = list(dataset_path.rglob('*.wav')) + \
                             list(dataset_path.rglob('*.flac')) + \
                             list(dataset_path.rglob('*.mp3'))

                size_mb = sum(f.stat().st_size for f in audio_files) / (1024 * 1024)

                print(f"\n[OK] {dataset}")
                print(f"   - 파일 수: {len(audio_files)}")
                print(f"   - 크기: {size_mb:.2f} MB")
                print(f"   - 경로: {dataset_path}")
            else:
                print(f"\n[ERROR] {dataset} - 없음")


def main():
    parser = argparse.ArgumentParser(description='오디오 데이터셋 다운로드')
    parser.add_argument(
        '--dataset',
        choices=['esc50', 'urbansound8k', 'librispeech', 'samples', 'all'],
        default='all',
        help='다운로드할 데이터셋 선택'
    )
    parser.add_argument(
        '--data-dir',
        default='data/datasets',
        help='데이터 저장 디렉토리'
    )

    args = parser.parse_args()

    downloader = DatasetDownloader(data_dir=args.data_dir)

    print("=" * 60)
    print("오디오 데이터셋 다운로더")
    print("=" * 60)
    print(f"저장 경로: {downloader.data_dir.absolute()}")

    if args.dataset == 'esc50' or args.dataset == 'all':
        downloader.download_esc50()

    if args.dataset == 'urbansound8k' or args.dataset == 'all':
        downloader.download_urbansound8k()

    if args.dataset == 'librispeech' or args.dataset == 'all':
        downloader.download_librispeech_test_clean()

    if args.dataset == 'samples' or args.dataset == 'all':
        downloader.download_sample_noises()

    # 다운로드된 데이터셋 정보 표시
    downloader.show_info()

    print("\n" + "=" * 60)
    print("다운로드 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
