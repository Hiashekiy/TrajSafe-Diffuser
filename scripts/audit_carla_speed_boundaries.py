"""Audit saved speed boundaries at actual trajectory adoption, not request time."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    directory = args.directory
    manifest = json.loads((directory/'scenario_manifest.json').read_text(encoding='utf8'))
    cfg = manifest['config']['safety']
    results = []
    for row in csv.DictReader((directory/'trajectory_decisions.csv').open(encoding='utf8')):
        if row['decision'] != 'ACCEPTED':
            continue
        boundary = json.loads(row['speed_boundary'])
        with np.load(directory/'candidates'/f"{row['source_frame']}.npz") as data:
            speed, s = data['reference_speed_mps'], data['arc_length_m']
            i = boundary['splice_index']
            acceleration = np.diff(speed[i:]**2)/(2*np.diff(s[i:]))
            results.append(dict(frame=int(row['frame']), source_frame=int(row['source_frame']),
                start_speed_mps=float(speed[i]), measured_start_speed_mps=boundary['measured_start_speed_mps'],
                initial_boundary_error_mps=abs(float(speed[i])-boundary['measured_start_speed_mps']),
                terminal_speed_mps=float(speed[-1]), min_acceleration_mps2=float(acceleration.min()),
                max_acceleration_mps2=float(acceleration.max())))
    report = dict(accepted_paths=len(results), boundaries=results,
        all_pass=bool(results) and all(r['initial_boundary_error_mps'] < 1e-6 and
            abs(r['terminal_speed_mps']) < 1e-6 and
            r['min_acceleration_mps2'] >= -cfg['comfort_decel_mps2']-1e-6 and
            r['max_acceleration_mps2'] <= cfg['max_accel_mps2']+1e-6 for r in results),
        scope='Scalar physical reference speeds at current-pose splice; actual speed is closed-loop tracking, not exact equality.')
    (directory/'speed_boundary_audit.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2))
    return 0 if report['all_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
