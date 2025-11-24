import sys
import os
from core import run

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run.py <file-path> [workdir]")
        sys.exit(1)

    fp = sys.argv[1]
    wd = sys.argv[2] if len(sys.argv) > 2 else None

    out = run(fp, workdir=wd)
    print(out)
