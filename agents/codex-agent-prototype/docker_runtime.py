"""One root Docker container for planner, checker and main controller roles."""
from __future__ import annotations

import argparse
import json
from urllib.parse import urlsplit, urlunsplit

from runtime_config import CONTAINER_BINARY, CONTAINER_CONFIG, RunPaths, controller_command


def docker_proxy(proxy: str) -> str:
    url = urlsplit(proxy)
    if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password:
        raise ValueError('proxy must be an HTTP(S) URL without embedded credentials')
    if url.hostname in ('localhost', '127.0.0.1', '::1'):
        host = 'host.docker.internal'
        if url.port:
            host += f':{url.port}'
        return urlunsplit((url.scheme, host, url.path, url.query, url.fragment))
    return proxy


def docker_command(args: argparse.Namespace, paths: RunPaths) -> list[str]:
    # Codex's read-only checker creates its own user/PID namespaces with bwrap.
    # Docker's default seccomp profile blocks that nested sandbox setup.
    cmd = ['docker', 'run', '--rm', '--init', '--user', '0:0',
           '--name', f'codex-{paths.root.name}', '--workdir', str(CONTAINER_CONFIG),
           '--add-host', 'host.docker.internal:host-gateway',
           '--security-opt', 'seccomp=unconfined']

    def mount(source: object, target: object, readonly: bool = False) -> None:
        # --mount avoids Docker silently creating missing host paths.
        value = f'type=bind,src={source},dst={target}'
        if ',' in str(source):
            raise ValueError('Docker bind paths cannot contain commas')
        cmd.extend(['--mount', value + (',readonly' if readonly else '')])

    mount(paths.workspace, '/home')
    mount(paths.output, '/run-output')
    mount(paths.inputs, '/run-input', True)
    mount((paths.inputs / 'smoke-skills') if args.smoke_test else (args.skills_root / args.paper_id / 'skill').resolve(), '/skills', True)
    if not args.smoke_test:
        mount(args.auth_dir.resolve(), '/run-auth')
    cmd += ['--env', 'HOME=/root', '--env', 'CODEX_HOME=/run-output/codex-home' if args.smoke_test else 'CODEX_HOME=/run-auth',
            '--env', 'CODEX_APP_SERVER_COMMAND=' + json.dumps([str(CONTAINER_BINARY)]),
            '--env', 'PYTHONUNBUFFERED=1']
    if args.proxy:
        proxy = docker_proxy(args.proxy)
        for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
            cmd += ['--env', f'{key}={proxy}']
        cmd += ['--env', 'NO_PROXY=localhost,127.0.0.1,::1', '--env', 'no_proxy=localhost,127.0.0.1,::1']
    for option in ('cpus', 'memory', 'gpus'):
        value = getattr(args, option)
        if value is not None:
            cmd += [f'--{option}', str(value)]
    if args.memory is not None:
        cmd += ['--memory-swap', str(args.memory)]
    cmd += ['--entrypoint', 'python3', args.image]
    cmd += controller_command(args, paths, docker=True)[1:]
    return cmd
