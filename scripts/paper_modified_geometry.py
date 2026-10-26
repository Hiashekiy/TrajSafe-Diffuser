import argparse,json,subprocess,sys
from pathlib import Path
ap=argparse.ArgumentParser(); ap.add_argument('--run',required=True); args=ap.parse_args()
run=Path(args.run); manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
subprocess.run([sys.executable,'-u',str(Path(__file__).with_name('compare_modified_geometry.py')),
    '--out',str(run/'geometry'),'--model-run',str(run/'A'),
    '--processed',manifest['processed_dataset']],check=True)
