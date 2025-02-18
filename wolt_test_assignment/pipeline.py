import subprocess
import sys


def run_command(command, description):
    print(f"Running: {description}...")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Error occurred during: {description}. Exiting...")
        sys.exit(result.returncode)
    print(f"Completed: {description}.\n")


try:
    run_command("python dataset.py", "Cleaning data")
    run_command("python features.py", "Preparing features and train/test split")
    run_command(
        "python modeling/train.py --epochs 20 --batch-size 16 --training-days 40 --n-steps 1",
        "Training models for next-day prediction",
    )
    run_command(
        "python modeling/train.py --epochs 60 --batch-size 32 --training-days 40 --n-steps 20",
        "Training models for multiple-day prediction",
    )
    run_command(
        "python modeling/eval_models_single_step.py --training-days 40",
        "Evaluating single-step models",
    )
    run_command(
        "python modeling/eval_models_multiple_step.py --training-days 40 --n-steps 20",
        "Evaluating multiple-step models",
    )
    print("All tasks completed successfully!")
except Exception as e:
    print(f"Pipeline failed: {e}")
