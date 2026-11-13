#!/usr/bin/env python3
"""Generate positive training examples for 'hey Joy' wake word."""

import os
import sys
from pathlib import Path

# Add synthetic speech repo to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'synthetic_speech_dataset_generation'))

def main():
    """Generate positive clips using both VITS and WAVEGLOW models."""
    import subprocess
    
    base_dir = Path(__file__).parent
    data_dir = base_dir / 'data'
    data_dir.mkdir(exist_ok=True)
    
    synth_repo = base_dir.parent / 'synthetic_speech_dataset_generation'
    os.chdir(synth_repo)
    
    python_exe = sys.executable
    
    # Generate with VITS model
    print("Generating 2500 clips with VITS...")
    vits_output = data_dir / 'positive_vits'
    vits_output.mkdir(exist_ok=True)
    subprocess.run([
        python_exe, 'generate_clips.py',
        '--model', 'VITS',
        '--enable_gpu',
        '--text', 'hey Joy',
        '--N', '2500',
        '--max_per_speaker', '1',
        '--output_dir', str(vits_output)
    ], check=True)
    
    # Generate with WAVEGLOW model
    print("Generating 2500 clips with WAVEGLOW...")
    waveglow_output = data_dir / 'positive_waveglow'
    waveglow_output.mkdir(exist_ok=True)
    subprocess.run([
        python_exe, 'generate_clips.py',
        '--model', 'WAVEGLOW',
        '--enable_gpu',
        '--text', 'hey Joy',
        '--N', '2500',
        '--max_per_speaker', '1',
        '--output_dir', str(waveglow_output)
    ], check=True)
    
    print("\nPositive data generation complete!")
    print(f"VITS clips: {vits_output}")
    print(f"WAVEGLOW clips: {waveglow_output}")

if __name__ == '__main__':
    main()
