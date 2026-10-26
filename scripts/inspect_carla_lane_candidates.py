"""Map inspection convenience entry point."""
import sys
from carla_online_closed_loop import main

if __name__ == '__main__':
    raise SystemExit(main(['--mode', 'inspect', *sys.argv[1:]]))
