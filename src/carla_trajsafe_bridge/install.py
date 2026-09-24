"""Install the CARLA bridge into a TrajSafe-Diffuser checkout.

    python install.py --repo E:/CarDataSample/Neural-IRISDiffuser

Copies src/carla_bridge, scripts/carla_*.py, configs/carla_demo.yaml and the
two docs.  Any file that already exists is first backed up next to itself with
a .bak suffix, so nothing in the model repo is ever lost silently.
"""

from __future__ import annotations

import argparse
import os
import shutil

HERE = os.path.abspath(os.path.dirname(__file__))

PLAN = (
    ("src/carla_bridge", "src/carla_bridge"),
    ("scripts/carla_trajsafe_demo.py", "scripts/carla_trajsafe_demo.py"),
    ("scripts/carla_verify_plan.py", "scripts/carla_verify_plan.py"),
    ("scripts/carla_trajsafe_continuous.py", "scripts/carla_trajsafe_continuous.py"),
    ("configs/carla_demo.yaml", "configs/carla_demo.yaml"),
    ("configs/carla_continuous.yaml", "configs/carla_continuous.yaml"),
    ("docs/CARLA_DEMO_TEST0056.md", "docs/CARLA_DEMO_TEST0056.md"),
    ("docs/CARLA_DEMO_TEST0056_SLIDE_TEXT.md", "docs/CARLA_DEMO_TEST0056_SLIDE_TEXT.md"),
)


def copy_tree(src, dst, installed):
    os.makedirs(dst, exist_ok=True)
    for name in sorted(os.listdir(src)):
        if name == "__pycache__":
            continue
        source = os.path.join(src, name)
        target = os.path.join(dst, name)
        if os.path.isdir(source):
            copy_tree(source, target, installed)
            continue
        if os.path.exists(target):
            shutil.copy2(target, target + ".bak")
        shutil.copy2(source, target)
        installed.append(target)


def copy_file(source, target, installed):
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if os.path.exists(target):
        shutil.copy2(target, target + ".bak")
    shutil.copy2(source, target)
    installed.append(target)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True,
                        help="target TrajSafe-Diffuser checkout")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    repo = os.path.abspath(args.repo)
    if not os.path.isdir(repo):
        raise SystemExit("target repo does not exist: %s" % repo)
    installed = []
    for source_rel, target_rel in PLAN:
        source = os.path.join(HERE, source_rel)
        target = os.path.join(repo, target_rel)
        if not os.path.exists(source):
            print("skip (not in package): %s" % source_rel)
            continue
        if args.dry_run:
            print("would install %s -> %s" % (source_rel, target))
            continue
        if os.path.isdir(source):
            copy_tree(source, target, installed)
        else:
            copy_file(source, target, installed)
    if args.dry_run:
        return 0
    for path in installed:
        print("installed %s" % path)
    print("%d files installed into %s" % (len(installed), repo))
    print("next: python scripts/carla_trajsafe_continuous.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
