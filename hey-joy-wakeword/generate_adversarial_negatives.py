#!/usr/bin/env python3
"""Generate adversarial negative examples - confusable phrases."""

import os
import sys
from pathlib import Path

# Add synthetic speech repo to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'synthetic_speech_dataset_generation'))

def main():
    """Generate adversarial negative clips with confusable phrases."""
    import subprocess
    
    base_dir = Path(__file__).parent
    data_dir = base_dir / 'data' / 'adversarial_negatives'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    synth_repo = base_dir.parent / 'synthetic_speech_dataset_generation'
    os.chdir(synth_repo)
    
    python_exe = sys.executable
    
    # Confusable phrases and their counts
    adversarial_phrases = [
        ("hey Joey", 2000),
        ("hey Joe", 1000),
        ("hey Joyce", 500),
        ("enjoy", 500),
        ("ahoy", 500),
        ("joy", 500),
        ("Joey", 500),
    ]
    
    for phrase, count in adversarial_phrases:
        print(f"\nGenerating {count} clips for '{phrase}'...")
        safe_name = phrase.replace(" ", "_")
        output_dir = data_dir / safe_name
        output_dir.mkdir(exist_ok=True)
        
        # Use VITS for half, WAVEGLOW for half
        half_count = count // 2
        
        # VITS
        print(f"  - {half_count} with VITS...")
        vits_dir = output_dir / 'vits'
        vits_dir.mkdir(exist_ok=True)
        subprocess.run([
            python_exe, 'generate_clips.py',
            '--model', 'VITS',
            '--enable_gpu',
            '--text', phrase,
            '--N', str(half_count),
            '--max_per_speaker', '1',
            '--output_dir', str(vits_dir)
        ], check=True)
        
        # WAVEGLOW
        print(f"  - {half_count} with WAVEGLOW...")
        waveglow_dir = output_dir / 'waveglow'
        waveglow_dir.mkdir(exist_ok=True)
        subprocess.run([
            python_exe, 'generate_clips.py',
            '--model', 'WAVEGLOW',
            '--enable_gpu',
            '--text', phrase,
            '--N', str(half_count),
            '--max_per_speaker', '1',
            '--output_dir', str(waveglow_dir)
        ], check=True)
    
    print("\nAdversarial negative generation complete!")
    print(f"Output: {data_dir}")

if __name__ == '__main__':
    main()
