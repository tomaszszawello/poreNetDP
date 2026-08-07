#!/usr/bin/env python3
""" Run __main__.py in parallel for a grid of Da_eff / cd_in parameters.

Why subprocesses, not threads or repeated imports
--------------------------------------------------
SimInputData in config.py is a class (not an instance), and several of its
attributes are *derived* from Da_eff/cd_in at class-body evaluation time
(Da, tmax, track_every, dirname, ...). That means:
  - Poking `SimInputData.Da_eff = x` after import does NOT recompute Da,
    tmax, dirname, etc. - they'd stay stale.
  - Running several parameter sets as threads in one interpreter would have
    them all fighting over the same shared class object.
  - Re-importing config.py in the same interpreter doesn't help either,
    since Python caches modules (sys.modules) and won't re-run it.

So each (Da_eff, cd_in) combination is run as its own `python __main__.py`
subprocess. Each subprocess gets its own fresh interpreter and its own fresh
import of config.py, parameterized via the SIM_DA_EFF / SIM_CD_IN
environment variables (see config.py), so every derived attribute is
computed correctly and independently, with no cross-talk between runs.

config.py's dirname now also encodes cd_in (previously only G/Da_eff), so
different cd_in values with the same Da_eff no longer collide on the same
output directory.

Usage
-----
python run_sweep.py
(edit DA_EFF_VALUES / CD_IN_VALUES / MAX_WORKERS below to customize the grid)
"""

import itertools
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(REPO_DIR, 'sweep_logs')

DA_EFF_VALUES = [0.2, 0.33]
CD_IN_VALUES = [0.33]
MAX_WORKERS = 20
"how many simulations to run at once - tune to available CPU/RAM"


def run_one(da_eff: float, cd_in: float) -> tuple[float, float, int]:
    """ Run one __main__.py subprocess with the given parameters.

    Parameters
    -------
    da_eff : float
        value to inject as SimInputData.Da_eff via SIM_DA_EFF env var

    cd_in : float
        value to inject as SimInputData.cd_in via SIM_CD_IN env var

    Returns
    -------
    tuple of (da_eff, cd_in, return code) of the finished subprocess
    """
    env = os.environ.copy()
    env['SIM_DA_EFF'] = str(da_eff)
    env['SIM_CD_IN'] = str(cd_in)
    # numpy/scipy here are linked against OpenBLAS, which by default spawns
    # its own thread pool per process. With MAX_WORKERS processes already
    # providing parallelism, letting each one also multithread its BLAS
    # calls oversubscribes the CPU (MAX_WORKERS * BLAS_threads vs. cpu_count
    # cores) and slows the whole sweep down. Pin each subprocess to 1 thread.
    for var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
        'NUMEXPR_NUM_THREADS'):
        env[var] = '1'

    tag = f'Daeff{da_eff:.3f}_cdin{cd_in:.3f}'
    log_path = os.path.join(LOG_DIR, f'{tag}.log')

    with open(log_path, 'w', encoding='utf-8') as log_file:
        log_file.write(f'# started {datetime.now().isoformat()}\n')
        log_file.flush()
        result = subprocess.run(
            [sys.executable, '__main__.py'],
            cwd=REPO_DIR,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

    status = 'OK' if result.returncode == 0 else f'FAILED (exit {result.returncode})'
    print(f'[{tag}] {status} -- log: {log_path}')
    return da_eff, cd_in, result.returncode


def main() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    combos = list(itertools.product(DA_EFF_VALUES, CD_IN_VALUES))
    if len(combos) != len(set(combos)):
        raise ValueError('Duplicate (Da_eff, cd_in) combos in the grid: '
            'utils.make_dir() is not safe against two processes racing to '
            'create the same output directory at the same time.')

    print(f'Running {len(combos)} simulations, {MAX_WORKERS} at a time.')
    print(f'Logs in {LOG_DIR}')

    failures = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for da_eff, cd_in, code in pool.map(lambda c: run_one(*c), combos):
            if code != 0:
                failures.append((da_eff, cd_in, code))

    if failures:
        print('\nSome runs failed:')
        for da_eff, cd_in, code in failures:
            print(f'  Da_eff={da_eff}, cd_in={cd_in} -> exit code {code}')
        sys.exit(1)

    print('\nAll runs completed successfully.')


if __name__ == '__main__':
    main()
