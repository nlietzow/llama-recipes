import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

if __name__ == '__main__':
    from custom.train_model import run_training

    run_training()
