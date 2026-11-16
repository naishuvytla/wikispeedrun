import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).parent.parent), str(Path(__file__).parent)]
