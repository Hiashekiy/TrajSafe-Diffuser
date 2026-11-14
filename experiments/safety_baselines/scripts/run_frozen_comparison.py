"""Run frozen S0/S1/S2/S3 in order; collision rates never gate progression."""
import datetime
import os
from pathlib import Path
import subprocess
import sys

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]

def main():
    tag = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    previous = None
    env = dict(os.environ)
    env.update(PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    for stage in ('S0', 'S1', 'S2', 'S3'):
        command = [sys.executable, '-u', str(BASE / 'run.py'), '--stage', stage]
        run_id = f'seven_frozen_t16_{stage}_{tag}'
        if stage != 'S0':
            command += ['--run-id', run_id]
            if previous:
                command += ['--previous', previous]
        print(f'START {stage}: {run_id}', flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        if stage != 'S0':
            previous = run_id
    print(f'COMPLETE: {previous}', flush=True)

if __name__ == '__main__':
    main()
