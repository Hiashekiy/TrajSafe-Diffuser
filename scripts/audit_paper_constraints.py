"""Audit every stored valid convex cell against full occupied pixel squares."""
import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.geometry.convex_region import halfspaces_to_vertices
from scripts.geometry_audit_utils import obstacle_boxes, audit_polygon

STATE = {}

def init(directory):
    for name in ('alm_cell_a', 'alm_cell_b', 'alm_cell_valid', 'occupancy'):
        STATE[name] = np.load(Path(directory)/(name+'.npy'), mmap_mode='r')

def audit(index):
    boxes, half = obstacle_boxes(np.asarray(STATE['occupancy'][index]).squeeze())
    counts = dict(regions=0, unsafe=0, invalid=0, outside=0)
    examples = []
    for cell in np.flatnonzero(STATE['alm_cell_valid'][index]):
        a, b = STATE['alm_cell_a'][index, cell], STATE['alm_cell_b'][index, cell]
        mask = np.isfinite(b) & (np.linalg.norm(a, axis=-1)>1e-8)
        r = audit_polygon(halfspaces_to_vertices(a[mask], b[mask]), boxes, half)
        counts['regions'] += 1
        counts['unsafe'] += int(r['colliding_cells']>0)
        counts['invalid'] += int(r['invalid'])
        counts['outside'] += int(r['outside'])
        if r['colliding_cells'] and len(examples)<3:
            examples.append(dict(index=index, cell=int(cell), overlap_area_m2=r['overlap_area_scene2']*6400))
    return counts, examples

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--processed', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--samples', type=int, default=0)
    args = ap.parse_args()
    report = {}
    for split in ('train', 'val', 'test'):
        directory = Path(args.processed)/split
        n = len(np.load(directory/'alm_cell_valid.npy', mmap_mode='r'))
        indices = range(n) if not args.samples else np.linspace(0,n-1,min(args.samples,n),dtype=int).tolist()
        result = dict(samples=len(indices), regions=0, unsafe=0, invalid=0, outside=0, examples=[])
        with mp.get_context('spawn').Pool(args.workers, initializer=init, initargs=(directory,)) as pool:
            for k, (counts, examples) in enumerate(pool.imap_unordered(audit, indices, chunksize=4), 1):
                for name, value in counts.items(): result[name] += value
                if len(result['examples'])<20: result['examples'].extend(examples)
                if k%500==0: print(split, k, '/',len(indices),result['unsafe'],flush=True)
        report[split] = result
        print(split, result, flush=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding='utf-8')
    if any(r['unsafe'] or r['invalid'] or r['outside'] for r in report.values()):
        raise RuntimeError('Offline constraint audit failed; training is blocked pending diagnosis')

if __name__=='__main__': main()
