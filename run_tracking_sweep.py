#!/usr/bin/env python3
""" Run tracking.py in parallel over every dissolution-simulation output
directory, i.e. every directory containing network_*.json snapshots.

Why subprocesses, and one subprocess per directory (not per network file)
---------------------------------------------------------------------
tracking.py's numba-jitted tracking functions get compiled once per
interpreter, so running many directories as threads in one interpreter
would serialize on the GIL during particle tracking anyway - subprocesses
give real parallelism, mirroring run_sweep.py.

Work is split per *directory*, not per network_*.json file, because all
snapshots in one directory share the same graph topology and the same b0
(aperture normalization) taken from the earliest snapshot - see
tracking.process_directory(). Splitting further would mean reloading/
renormalizing against a different b0 per file, changing results.

G and Da for each directory are recovered from the 'G<G>Daeff<Da_eff>'
segment that config.py bakes into the simulation's dirname, converting
Da_eff -> Da the same way config.py does (Da = Da_eff * (1 + G)) - see
tracking.parse_g_da_from_dirname().

Usage
-----
python run_tracking_sweep.py
(edit ROOT_DIRS / MAX_WORKERS / N_PARTS below to customize)
"""

import itertools
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(REPO_DIR, 'sweep_logs_tracking')

ROOT_DIRS = ['dfn_het_apertures']
"top-level dirs (relative to REPO_DIR) to search for network_*.json files"
MAX_WORKERS = 20
"how many directories to track at once - tune to available CPU/RAM"
N_PARTS = 100000
"number of particles to track per network snapshot (TRACK_N_PARTS)"

SKIP_COMPLETED = True
"skip directories that already finished successfully in a previous run (checks for .done marker)"


def find_network_dirs() -> list[str]:
    """ Find every directory under ROOT_DIRS that contains network_*.json
    files, as paths relative to REPO_DIR.
    """
    dirs = []
    for root_dir in ROOT_DIRS:
        for dirpath, _, filenames in os.walk(os.path.join(REPO_DIR, root_dir)):
            if any(f.startswith('network_') and f.endswith('.json') for f in filenames):
                dirs.append(os.path.relpath(dirpath, REPO_DIR))
    return sorted(dirs)


def _tag(dirname: str) -> str:
    return dirname.replace('/', '_')


def _done_path(tag: str) -> str:
    return os.path.join(LOG_DIR, f'{tag}.done')


def _is_done(tag: str) -> bool:
    return SKIP_COMPLETED and os.path.exists(_done_path(tag))


def _mark_done(tag: str) -> None:
    with open(_done_path(tag), 'w', encoding='utf-8') as f:
        f.write(datetime.now().isoformat() + '\n')


def run_one(dirname: str) -> tuple[str, int]:
    """ Run tracking.py as a subprocess for every network_*.json file in
    `dirname` (a path relative to REPO_DIR).

    Returns
    -------
    tuple of (dirname, return code) of the finished subprocess
    """
    env = os.environ.copy()
    env['TRACK_DIRNAME'] = dirname + '/'
    env['TRACK_N_PARTS'] = str(N_PARTS)
    # see run_sweep.py's run_one() for why we pin BLAS/OMP to 1 thread here:
    # MAX_WORKERS subprocesses already provide the parallelism.
    for var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
        'NUMEXPR_NUM_THREADS'):
        env[var] = '1'

    tag = _tag(dirname)
    log_path = os.path.join(LOG_DIR, f'{tag}.log')

    if _is_done(tag):
        print(f'[{tag}] SKIPPED (completed in a previous run)')
        return dirname, 0

    with open(log_path, 'w', encoding='utf-8') as log_file:
        log_file.write(f'# started {datetime.now().isoformat()}\n')
        log_file.flush()
        result = subprocess.run(
            [sys.executable, 'tracking.py'],
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
    return dirname, result.returncode


def main() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    dirs = find_network_dirs()
    if not dirs:
        raise ValueError(
            f'No network_*.json files found under {ROOT_DIRS} - nothing to track.')

    print(f'Tracking {len(dirs)} directories, {MAX_WORKERS} at a time.')
    print(f'Logs in {LOG_DIR}')

    failures = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for dirname, code in pool.map(run_one, dirs):
            if code != 0:
                failures.append((dirname, code))

    if failures:
        print('\nSome runs failed:')
        for dirname, code in failures:
            print(f'  {dirname} -> exit code {code}')
        sys.exit(1)

    print('\nAll runs completed successfully.')


if __name__ == '__main__':
    main()
