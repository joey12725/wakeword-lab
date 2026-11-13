#!/usr/bin/env python3
"""Download standard negative training data."""

import urllib.request
import zipfile
from pathlib import Path
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    """Download a file with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def download_and_extract(url, data_dir, extract_dir_name):
    """Download and extract a zip file."""
    zip_path = data_dir / f"{extract_dir_name}.zip"
    extract_path = data_dir / extract_dir_name
    
    if extract_path.exists():
        print(f"\n{extract_dir_name} already exists, skipping...")
        return extract_path
    
    print(f"\nDownloading {extract_dir_name}...")
    download_url(url, zip_path)
    
    print(f"Extracting {extract_dir_name}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    # Clean up zip file
    zip_path.unlink()
    
    return extract_path

def download_common_voice_samples():
    """Download Common Voice samples via HuggingFace."""
    from datasets import load_dataset
    
    print("\nDownloading 10,000 Common Voice 11 samples...")
    
    # Load dataset
    ds = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="train", streaming=True)
    
    # Create output directory
    cv_dir = Path(__file__).parent / 'data' / 'common_voice_samples'
    cv_dir.mkdir(parents=True, exist_ok=True)
    
    # Download 10,000 samples
    import soundfile as sf
    count = 0
    for i, example in enumerate(ds):
        if count >= 10000:
            break
        
        # Save audio
        audio_path = cv_dir / f"cv_{i:06d}.wav"
        audio_data = example['audio']['array']
        sample_rate = example['audio']['sampling_rate']
        
        sf.write(str(audio_path), audio_data, sample_rate)
        count += 1
        
        if count % 100 == 0:
            print(f"  Downloaded {count}/10000 samples...")
    
    print(f"Common Voice samples saved to {cv_dir}")
    return cv_dir

def main():
    """Download all negative training data."""
    base_dir = Path(__file__).parent
    data_dir = base_dir / 'data' / 'negatives'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download FMA sample
    fma_url = "https://f002.backblazeb2.com/file/openwakeword-resources/data/fma_sample.zip"
    download_and_extract(fma_url, data_dir, "fma_sample")
    
    # Download FSD50k sample
    fsd50k_url = "https://f002.backblazeb2.com/file/openwakeword-resources/data/fsd50k_sample.zip"
    download_and_extract(fsd50k_url, data_dir, "fsd50k_sample")
    
    # Download Common Voice samples
    try:
        # Try soundfile first (needed for Common Voice)
        import soundfile
        download_common_voice_samples()
    except ImportError:
        print("\nsoundfile not installed - skipping Common Voice download")
        print("Install with: pip install soundfile")
    
    print("\n✓ All negative data downloaded successfully!")

if __name__ == '__main__':
    main()
