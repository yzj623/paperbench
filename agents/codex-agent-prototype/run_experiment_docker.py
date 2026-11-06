#!/usr/bin/env python3
"""Docker-default entrypoint; all roles run in the same controller container."""
from run_paperbench import main

if __name__ == '__main__':
    raise SystemExit(main(docker_default=True))
