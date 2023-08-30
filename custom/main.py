import sys
from pathlib import Path

from huggingface_hub import login

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

if __name__ == '__main__':
    from custom.eval_model import eval_model

    login()
    eval_model()
