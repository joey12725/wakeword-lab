#!/usr/bin/env python3
"""Master script to orchestrate the complete training pipeline."""

import subprocess
import sys
from pathlib import Path
import time

def run_step(step_name, script_path, log_dir):
    """Run a training step and log output."""
    print(f"\n{'='*80}")
    print(f"STEP: {step_name}")
    print(f"{'='*80}\n")
    
    log_file = log_dir / f"{step_name.lower().replace(' ', '_')}.log"
    
    start_time = time.time()
    
    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # Print and log output
            output = result.stdout
            print(output)
            f.write(output)
            
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, result.args, output)
        
        elapsed = time.time() - start_time
        print(f"\n✓ {step_name} completed in {elapsed/60:.1f} minutes")
        return True
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ {step_name} failed after {elapsed/60:.1f} minutes: {e}")
        print(f"Check log: {log_file}")
        return False

def main():
    """Run the complete training pipeline."""
    base_dir = Path(__file__).parent
    log_dir = base_dir / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("HEY JOY WAKE WORD MODEL TRAINING")
    print("="*80)
    
    start_time = time.time()
    
    # Step 1: Generate positive data
    if not run_step(
        "Generate Positive Data",
        base_dir / "generate_positives.py",
        log_dir
    ):
        return 1
    
    # Step 2: Generate adversarial negatives
    if not run_step(
        "Generate Adversarial Negatives",
        base_dir / "generate_adversarial_negatives.py",
        log_dir
    ):
        return 1
    
    # Step 3: Download standard negatives
    if not run_step(
        "Download Standard Negatives",
        base_dir / "download_negatives.py",
        log_dir
    ):
        return 1
    
    # Step 4: Train the model
    if not run_step(
        "Train Model",
        base_dir / "train_model.py",
        log_dir
    ):
        return 1
    
    # Complete!
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print(f"✓ TRAINING COMPLETE!")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Model output: {base_dir / 'models' / 'hey_joy.onnx'}")
    print("="*80)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
