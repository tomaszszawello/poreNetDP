#!/usr/bin/env python3
""" Run __main__.py in parallel for a grid of load_name / Da_eff / G parameters.

Why subprocesses, not threads or repeated imports
--------------------------------------------------
SimInputData in config.py is a class (not an instance), and several of its
attributes are *derived* from load_name/Da_eff/G at class-body evaluation
time (Da, dirname, ...). That means:
  - Poking `SimInputData.Da_eff = x` after import does NOT recompute Da,
    dirname, etc. - they'd stay stale.
  - Running several parameter sets as threads in one interpreter would have
    them all fighting over the same shared class object.
  - Re-importing config.py in the same interpreter doesn't help either,
    since Python caches modules (sys.modules) and won't re-run it.

So each (load_name, Da_eff, G) combination is run as its own
`python __main__.py` subprocess. Each subprocess gets its own fresh
interpreter and its own fresh import of config.py, parameterized via the
SIM_LOAD_NAME / SIM_DA_EFF / SIM_G environment variables (see config.py),
so every derived attribute is computed correctly and independently, with no
cross-talk between runs.

config.py's dirname encodes load_name/G/Da_eff, so different combos never
collide on the same output directory.

Usage
-----
python run_sweep.py
(edit LOAD_NAME_VALUES / DA_EFF_VALUES / G_VALUES / MAX_WORKERS below to
customize the grid)
"""

import glob
import itertools
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(REPO_DIR, 'sweep_logs3')

LOAD_NAME_VALUES = sorted(
    os.path.relpath(path, REPO_DIR)[:-len('.json')]
    for path in glob.glob(os.path.join(REPO_DIR, 'dfn_het_apertures', 'sigma_*.json'))
)
"load_name (network file, minus .json) for every sigma_*.json in dfn_het_apertures/"
DA_EFF_VALUES = [0.002, 0.02, 0.2]
G_VALUES = [0.1, 1., 5.]
MAX_WORKERS = 20
"how many simulations to run at once - tune to available CPU/RAM"

SKIP_COMPLETED = True
"skip combos that already finished successfully in a previous run (checks for .done marker)"


def _tag(load_name: str, da_eff: float, g: float) -> str:
    load_tag = load_name.replace('/', '_')
    return f'{load_tag}_G{g:.3f}_Daeff{da_eff:.3f}'


def _done_path(tag: str) -> str:
    return os.path.join(LOG_DIR, f'{tag}.done')


def _is_done(tag: str) -> bool:
    return SKIP_COMPLETED and os.path.exists(_done_path(tag))


def _mark_done(tag: str) -> None:
    with open(_done_path(tag), 'w', encoding='utf-8') as f:
        f.write(datetime.now().isoformat() + '\n')


def run_one(load_name: str, da_eff: float, g: float) -> tuple[str, float, float, int]:
    """ Run one __main__.py subprocess with the given parameters.

    Parameters
    -------
    load_name : str
        value to inject as SimInputData.load_name via SIM_LOAD_NAME env var

    da_eff : float
        value to inject as SimInputData.Da_eff via SIM_DA_EFF env var

    g : float
        value to inject as SimInputData.G via SIM_G env var

    Returns
    -------
    tuple of (load_name, da_eff, g, return code) of the finished subprocess
    """
    env = os.environ.copy()
    env['SIM_LOAD_NAME'] = load_name
    env['SIM_DA_EFF'] = str(da_eff)
    env['SIM_G'] = str(g)
    # numpy/scipy here are linked against OpenBLAS, which by default spawns
    # its own thread pool per process. With MAX_WORKERS processes already
    # providing parallelism, letting each one also multithread its BLAS
    # calls oversubscribes the CPU (MAX_WORKERS * BLAS_threads vs. cpu_count
    # cores) and slows the whole sweep down. Pin each subprocess to 1 thread.
    for var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
        'NUMEXPR_NUM_THREADS'):
        env[var] = '1'

    tag = _tag(load_name, da_eff, g)
    log_path = os.path.join(LOG_DIR, f'{tag}.log')

    if _is_done(tag):
        print(f'[{tag}] SKIPPED (completed in a previous run)')
        return load_name, da_eff, g, 0

    with open(log_path, 'w', encoding='utf-8') as log_file:
        log_file.write(f'# started {datetime.now().isoformat()}\n')
        log_file.flush()
        result = subprocess.run(
            [sys.executable, '__main__.py'],
            cwd=REPO_DIR,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )

    if result.returncode == 0:
        _mark_done(tag)
        print(f'[{tag}] OK -- log: {log_path}')
    else:
        print(f'[{tag}] FAILED (exit {result.returncode}) -- log: {log_path}')
    return load_name, da_eff, g, result.returncode


def main() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    if not LOAD_NAME_VALUES:
        raise ValueError(
            'No sigma_*.json files found in dfn_het_apertures/ - nothing to sweep.')
    combos = list(itertools.product(LOAD_NAME_VALUES, DA_EFF_VALUES, G_VALUES))
    if len(combos) != len(set(combos)):
        raise ValueError('Duplicate (load_name, Da_eff, G) combos in the grid: '
            'utils.make_dir() is not safe against two processes racing to '
            'create the same output directory at the same time.')

    print(f'Running {len(combos)} simulations, {MAX_WORKERS} at a time.')
    print(f'Logs in {LOG_DIR}')

    failures = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for load_name, da_eff, g, code in pool.map(lambda c: run_one(*c), combos):
            if code != 0:
                failures.append((load_name, da_eff, g, code))

    if failures:
        print('\nSome runs failed:')
        for load_name, da_eff, g, code in failures:
            print(f'  load_name={load_name}, Da_eff={da_eff}, G={g} -> exit code {code}')
        sys.exit(1)

    print('\nAll runs completed successfully.')


if __name__ == '__main__':
    main()
