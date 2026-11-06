"""Local-only launch configuration. No credential contents are inspected."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CONTAINER_CONFIG = Path('/opt/codex-config')
CONTAINER_BINARY = Path('/opt/codex/bin/codex-app-server')


def positive_seconds(value: str) -> float:
    seconds = float(value)
    if seconds <= 0 or not seconds < float('inf'):
        raise argparse.ArgumentTypeError('timeout must be finite and positive')
    return seconds


def positive_timeout(value: str) -> int:
    seconds = positive_seconds(value)
    if not seconds.is_integer():
        raise argparse.ArgumentTypeError('controller timeout must be whole seconds')
    return int(seconds)


def parser(*, docker_default: bool = True) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Launch the local Codex PaperBench controller')
    p.add_argument('--paper-id', required=True)
    p.add_argument('--runtime', choices=('host', 'docker'), default='docker' if docker_default else 'host')
    p.add_argument('--image', default='codex-paperbench:latest')
    p.add_argument('--runs-dir', type=Path, default=PROJECT_ROOT / 'runs')
    p.add_argument('--auth-dir', type=Path, default=PROJECT_ROOT / '.codex-auth')
    p.add_argument('--skills-root', type=Path, default=PROJECT_ROOT / 'skills')
    p.add_argument('--paperbench-root', type=Path, default=PROJECT_ROOT / 'vendor/paperbench-official')
    p.add_argument('--app-server', type=Path, default=PROJECT_ROOT / 'build/bin/codex-app-server')
    p.add_argument('--model', default='gpt-6-luna', help='Model ID; defaults to gpt-6-luna')
    p.add_argument('--ra-file-tools', action='store_true',
                   help='Enable RA main-agent file tools in the separate Codex variant')
    p.add_argument('--strip-irrelevant-developer-context', action=argparse.BooleanOptionalAction, default=False,
                   help='Disable Codex native multi-agent, plugin/app and collaboration developer context '
                        '(also disables the corresponding tools and plugin skills)')
    p.add_argument('--checker', action=argparse.BooleanOptionalAction, default=True,
                   help='Enable the semantic Check Agent (default: enabled); use --no-checker to auto-accept subtasks')
    p.add_argument('--planner-timeout', type=positive_timeout, default=None,
                   help='Optional planner timeout; omitted by default')
    p.add_argument('--checker-timeout', type=positive_timeout, default=None,
                   help='Optional checker timeout; omitted by default')
    p.add_argument('--main-timeout', type=positive_timeout, default=None)
    p.add_argument('--plan', type=Path, help='Use a fixed plan (Docker path: /run-input/plan.json)')
    p.add_argument('--proxy', help='Optional HTTP(S) proxy; Docker loopback maps to host.docker.internal (Clash host)')
    p.add_argument('--cpus', type=positive_seconds, help='Optional Docker CPU limit; no default limit')
    p.add_argument('--memory', help='Optional Docker memory limit; no default limit')
    p.add_argument('--gpus', help='Optional Docker --gpus value')
    p.add_argument('--smoke-test', action='store_true', help='Controller smoke test without auth or inference')
    p.add_argument('--dry-run', action='store_true', help='Print launch command without creating a run or checking auth')
    p.add_argument('--no-grade', action='store_true', help='Skip local grading after successful final submission')
    return p


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def workspace(self) -> Path:
        return self.root / 'home'

    @property
    def output(self) -> Path:
        return self.root / 'output'

    @property
    def inputs(self) -> Path:
        return self.root / 'input'


def validate_paper_id(paper_id: str) -> None:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', paper_id):
        raise ValueError('paper-id must be a simple identifier, without path separators')


def controller_command(args: argparse.Namespace, paths: RunPaths, *, docker: bool) -> list[str]:
    config = CONTAINER_CONFIG if docker else PROJECT_ROOT
    command = ['python3' if docker else sys.executable, str(config / 'controller.py'),
               '--paper-id', args.paper_id,
               '--workspace', '/home' if docker else str(paths.workspace),
               '--output', '/run-output' if docker else str(paths.output),
               '--skills', '/skills' if docker else str(paths.inputs / 'smoke-skills' if args.smoke_test else (args.skills_root / args.paper_id / 'skill').resolve()),
               '--codex-home', '/run-output/codex-home' if docker and args.smoke_test else '/run-auth' if docker else str(paths.root / 'codex-home' if args.smoke_test else args.auth_dir.resolve()),
               '--plan-input', '/run-input/plan.json' if docker else str(paths.inputs / 'plan.json')]
    if args.planner_timeout is not None:
        command += ['--planner-timeout', str(args.planner_timeout)]
    if args.checker_timeout is not None:
        command += ['--checker-timeout', str(args.checker_timeout)]
    if not args.checker:
        command += ['--no-checker']
    if args.main_timeout is not None:
        command += ['--main-timeout', str(args.main_timeout)]
    if args.model:
        command += ['--model', args.model]
    if args.ra_file_tools:
        command += ['--ra-file-tools']
    if args.strip_irrelevant_developer_context:
        command += ['--strip-irrelevant-developer-context']
    if args.smoke_test:
        command += ['--smoke-test']
    return command


def agent_environment(args: argparse.Namespace, paths: RunPaths) -> dict[str, str]:
    # Do not inherit agent API keys, personal Codex settings, or grader secrets.
    allowed = ('PATH', 'LANG', 'LC_ALL', 'TERM', 'SSL_CERT_FILE', 'SSL_CERT_DIR',
               'CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH')
    env = {key: os.environ[key] for key in allowed if key in os.environ}
    env.update(HOME=str(paths.root / 'host-home'),
               CODEX_HOME=str(paths.root / 'codex-home') if args.smoke_test else str(args.auth_dir.resolve()),
               PYTHONUNBUFFERED='1', PYTHONNOUSERSITE='1')
    env['CODEX_APP_SERVER_COMMAND'] = json.dumps([str(args.app_server.resolve())])
    if args.proxy:
        env.update({key: args.proxy for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy')})
    return env
