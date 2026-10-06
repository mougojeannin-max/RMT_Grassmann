"""Independent command-line review: run experiments, then compare references.

Uses only the standard library and imports no experimental implementation.
Run in the Python environment installed from requirements.txt.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--quick', action='store_true', help='Tests and two simulation draws; no real data')
    mode.add_argument('--full-grid', action='store_true', help='Also repeat all hyperparameter searches (very expensive)')
    parser.add_argument('--results', type=Path, default=Path('results'))
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--distance-threads', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1 or args.distance_threads < 1:
        parser.error('Worker and thread counts must be positive')
    folder = args.results if args.results.is_absolute() else ROOT / args.results
    folder.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.setdefault('MPLCONFIGDIR', str(folder / '.matplotlib'))
    report_path = folder / ('quick_check.json' if args.quick else 'reproduction_check.json')
    report = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                  scope='quick' if args.quick else ('full_grid' if args.full_grid else 'selected_parameters'),
                  complete_reproduction=False, hyperparameter_search_checked=False, steps=[])

    def save():
        report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')

    def run(*command):
        command = [str(value) for value in command]
        step = dict(command=command, status='running')
        report['steps'].append(step)
        save()
        print('\n>>> python ' + ' '.join(command), flush=True)
        start = time.monotonic()
        try:
            subprocess.run([sys.executable, *command], cwd=ROOT, env=environment, check=True)
        except BaseException:
            step['status'] = 'failed_or_interrupted'
            raise
        else:
            step['status'] = 'passed'
        finally:
            step['seconds'] = round(time.monotonic() - start, 3)
            save()

    save()
    try:
        run('-m', 'pip', 'check')
        run('-m', 'unittest', 'discover', '-s', 'tests', '-v')
        if args.quick:
            simulation = folder / 'check_smoke'
            run('simulation.py', '--dimensions', 32, '--repetitions', 2, '--output', simulation)
            run('verify.py', '--simulation', simulation, '--allow-partial',
                '--output', folder / 'quick_comparison.json')
        else:
            run('verify.py', '--data', 'data', '--output', folder / 'data_comparison.json')
            run('figure1.py', '--output', folder / 'figure1')
            run('simulation.py', '--workers', args.workers, '--output', folder / 'simulation')
            tables = folder / ('real_data_full_grid' if args.full_grid else 'real_data')
            command = ['real_data.py', '--output', tables, '--workers', args.workers,
                       '--distance-threads', args.distance_threads]
            if not args.full_grid:
                command += ['--selected']
            if (tables / 'protocol.json').exists():
                command.append('--resume')
            run(*command)
            command = ['verify.py', '--figure1', folder / 'figure1',
                       '--simulation', folder / 'simulation', '--tables', tables,
                       '--output', folder / 'numerical_comparison.json']
            if args.full_grid:
                command += ['--selection', tables]
            run(*command)
            report['complete_reproduction'] = True
            report['hyperparameter_search_checked'] = args.full_grid
        report['status'] = 'passed'
    except (subprocess.CalledProcessError, OSError, KeyboardInterrupt) as error:
        report['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        report['error'] = str(error)
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        save()
        print('\nReview report: ' + str(report_path), flush=True)
    if args.quick:
        print('Quick check passed. Figure 1 and real-data tables remain unchecked.')
    else:
        print('Figure 1, both simulation plots and all 3,240 test scores passed.')
        if not args.full_grid:
            print('Hyperparameter searches were not repeated; use --full-grid to check them.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
