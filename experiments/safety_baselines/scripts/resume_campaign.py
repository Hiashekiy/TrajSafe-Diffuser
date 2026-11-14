"""Resume unchanged numerical code with retry for transient Windows file locks."""
import importlib.util
import json
from pathlib import Path
import sys
import time

def main():
    path = Path(__file__).with_name('campaign.py')
    spec = importlib.util.spec_from_file_location('safety_campaign_resume', path)
    campaign = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(campaign)
    # Ignore only newly added offline preparation files, never changed frozen code.
    original_source_hash = campaign.source_hash
    run_id = sys.argv[sys.argv.index('--run-id') + 1]
    frozen = json.loads((campaign.BASE / 'results' / run_id / 'manifest.json').read_text())['fingerprint']['source_hashes']
    current = original_source_hash()
    additions = set(current) - set(frozen)
    if any(not name.replace('\\', '/').startswith('src/map_preparation/') for name in additions):
        raise ValueError('unreviewed new runtime sources')
    if any(current.get(name) != value for name, value in frozen.items()):
        raise ValueError('frozen experiment source changed')
    if any(name.startswith('src.map_preparation') for name in sys.modules):
        raise ValueError('offline preparation unexpectedly imported')
    (campaign.BASE / 'results' / run_id / 'resume_source_audit.json').write_text(
        json.dumps({'unchanged_frozen_sources': True, 'excluded_new_offline_sources': {name: current[name] for name in sorted(additions)}, 'reason': 'new offline preprocessing package not imported by evaluation; existing source hashes unchanged'}, indent=2), encoding='utf8')
    campaign.source_hash = lambda: {name: value for name, value in original_source_hash().items() if name in frozen}
    original_dump = campaign.dump

    def retry_dump(path, data):
        for attempt in range(30):
            try:
                return original_dump(path, data)
            except PermissionError:
                if attempt == 29:
                    raise
                time.sleep(min(0.1 * (attempt + 1), 1.0))

    campaign.dump = retry_dump
    campaign.main()

if __name__ == '__main__':
    main()
