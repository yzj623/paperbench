#!/usr/bin/env python3
"""Independent host/Docker launch and post-controller local grading."""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

from docker_runtime import docker_command
from runtime_config import PROJECT_ROOT, RunPaths, agent_environment, controller_command, parser, validate_paper_id


def prepare_workspace(args, paths: RunPaths) -> None:
    for folder in (paths.workspace, paths.output, paths.inputs, paths.root / 'host-home', paths.root / 'codex-home'):
        folder.mkdir()
    (paths.workspace / 'submission').mkdir()
    public = paths.workspace / 'paper'
    public.mkdir()
    if not args.smoke_test:
        source = args.paperbench_root.resolve() / 'project/paperbench/data/papers' / args.paper_id
        # Explicit public allowlist: never expose rubric, reference, or grading files.
        for name in ('paper.pdf', 'paper.md', 'addendum.md', 'assets', 'blacklist.txt'):
            item = source / name
            if item.is_dir():
                shutil.copytree(item, public / name)
            elif item.is_file():
                shutil.copy2(item, public / name)
        instructions = args.paperbench_root.resolve() / 'project/paperbench/paperbench/instructions/code_only_instructions.txt'
        shutil.copy2(instructions, paths.workspace / 'instructions.txt')
    else:
        (public / 'paper.md').write_text('# Offline launcher smoke fixture\nImplement a deterministic identity function.\n')
        (public / 'addendum.md').write_text('No inference or external requests are required.\n')
        (public / 'blacklist.txt').write_text('')
        (paths.workspace / 'instructions.txt').write_text('Offline smoke test: create README.md and reproduce.sh under submission.\n')
        smoke_skill = paths.inputs / 'smoke-skills' / 'offline-smoke'
        smoke_skill.mkdir(parents=True)
        (smoke_skill / 'SKILL.md').write_text('---\nname: offline-smoke\ndescription: Offline smoke fixture\n---\nUse deterministic local checks only.\n')
    subprocess.run(['git', 'init', '--quiet', str(paths.workspace / 'submission')], check=True)
    for key, value in (('user.name', 'PaperBench Agent'), ('user.email', 'agent@localhost')):
        subprocess.run(['git', '-C', str(paths.workspace / 'submission'), 'config', key, value], check=True)
    workspace = '/home' if args.runtime == 'docker' else str(paths.workspace)
    (paths.workspace / 'agent.env').write_text(
        f'export UV_PROJECT_ENVIRONMENT={shlex.quote(workspace + "/.venv")}\n')
    hook = paths.workspace / 'submission/.git/hooks/pre-commit'
    # PaperBench's 1 GiB tracked-file limit, with whitespace-safe filenames.
    hook.write_text('#!/usr/bin/env bash\nset -euo pipefail\ntotal=0\n'
                    'while IFS= read -r -d "" file; do\n'
                    '  if [[ -f "$file" ]]; then\n'
                    '    size=$(stat -c%s -- "$file")\n'
                    '    total=$((total + size))\n'
                    '  fi\n'
                    'done < <(git ls-files -z)\n'
                    'if (( total > 1024 * 1024 * 1024 )); then\n'
                    '  echo "Tracked submission files exceed the 1 GiB limit" >&2\n'
                    '  exit 1\nfi\n')
    hook.chmod(0o755)
    if not args.smoke_test:
        solvers = args.paperbench_root.resolve() / 'project/paperbench/paperbench/solvers'
        # Only public solver assets/templates, never solvers/agent.env or .env.
        example = solvers / 'agent.env.example'
        if example.is_file():
            shutil.copy2(example, paths.workspace / 'agent.env')
        original_hook = solvers / 'pre-commit'
        if original_hook.is_file():
            # Official checkout may use CRLF; normalize its shell script.
            hook.write_text(original_hook.read_text(encoding='utf-8'))
            hook.chmod(0o755)
        launch_config = solvers / 'launch.json'
        if launch_config.is_file():
            (paths.workspace / '.vscode').mkdir()
            shutil.copy2(launch_config, paths.workspace / '.vscode/launch.json')
    if args.plan:
        shutil.copy2(args.plan.resolve(), paths.inputs / 'plan.json')


def frozen_submission(paths: RunPaths) -> Path:
    """Grade only the controller-owned accepted snapshot, never the live tree."""
    status = json.loads((paths.output / 'status.json').read_text(encoding='utf-8'))
    if status.get('phase') != 'complete' or status.get('accepted') is not True:
        raise RuntimeError('controller status does not confirm accepted completion')
    events = paths.output / 'protocol/tool_calls.jsonl'
    if not events.is_file():
        raise RuntimeError('controller did not record a successful final_submit')
    accepted = False
    for line in events.read_text(encoding='utf-8').splitlines():
        record = json.loads(line)
        result = record.get('result', {})
        if record.get('event') == 'tool_result' and record.get('tool') == 'final_submit' and isinstance(result, dict) and result.get('ok') is True:
            accepted = True
    if not accepted:
        raise RuntimeError('controller did not record a successful final_submit')
    snapshot = paths.output / 'frozen/submission'
    if not snapshot.is_dir():
        raise RuntimeError('controller frozen/submission directory is missing')
    return snapshot


def host_status(paths: RunPaths, phase: str, **fields) -> None:
    value = {'phase': phase, 'updated_at': datetime.now(timezone.utc).isoformat(), **fields}
    temporary = paths.root / '.host-status.tmp'
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(paths.root / 'host_status.json')


def launch(args) -> int:
    validate_paper_id(args.paper_id)
    args.runs_dir = args.runs_dir.resolve()
    if args.proxy:
        from docker_runtime import docker_proxy
        docker_proxy(args.proxy)  # Validate even for host/dry-run, never log proxy credentials.
    if args.dry_run:
        paths = RunPaths(args.runs_dir / f'{args.paper_id}-DRY-RUN')
        command = docker_command(args, paths) if args.runtime == 'docker' else controller_command(args, paths, docker=False)
        print(shlex.join(command))
        return 0
    if not (PROJECT_ROOT / 'controller.py').is_file():
        raise FileNotFoundError('local controller.py is required')
    if not args.smoke_test:
        if not args.auth_dir.resolve().is_dir():
            raise FileNotFoundError('provide a project .codex-auth directory or --auth-dir; launcher never logs in')
        if not (args.skills_root.resolve() / args.paper_id / 'skill').is_dir():
            raise FileNotFoundError('complete local skills/<paper-id>/skill directory is required')
        paper = args.paperbench_root.resolve() / 'project/paperbench/data/papers' / args.paper_id
        if not (paper / 'paper.md').is_file() and not (paper / 'paper.pdf').is_file():
            raise FileNotFoundError(f'public paper not found: {args.paper_id}')
    if args.plan and not args.plan.is_file():
        raise FileNotFoundError('fixed plan must be a local file')
    if args.runtime == 'host' and not args.smoke_test and not args.app_server.is_file():
        raise FileNotFoundError('build the local standalone app-server first with build_codex.sh')
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    paths = RunPaths(Path(tempfile.mkdtemp(prefix=f'{args.paper_id}-', dir=args.runs_dir)))
    host_status(paths, 'preparing', pid=os.getpid(), paper_id=args.paper_id, runtime=args.runtime)
    try:
        prepare_workspace(args, paths)
    except Exception:
        host_status(paths, 'preparation_failed')
        raise
    command = docker_command(args, paths) if args.runtime == 'docker' else controller_command(args, paths, docker=False)
    print(f'Run directory: {paths.root}', flush=True)
    env = agent_environment(args, paths)
    metadata = {'paper_id': args.paper_id, 'runtime': args.runtime,
                'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'skill_count': sum(1 for _ in ((paths.inputs / 'smoke-skills') if args.smoke_test else
                                  (args.skills_root / args.paper_id / 'skill')).rglob('SKILL.md')),
                'workspace_host_path': str(paths.workspace), 'output_host_path': str(paths.output),
                'image': args.image if args.runtime == 'docker' else None,
                'container_name': f'codex-{paths.root.name}' if args.runtime == 'docker' else None,
                'planner_timeout': args.planner_timeout, 'checker_timeout': args.checker_timeout,
                'main_timeout': args.main_timeout, 'model': args.model, 'checker': args.checker,
                'strip_irrelevant_developer_context': args.strip_irrelevant_developer_context,
                'smoke_test': args.smoke_test}
    (paths.root / 'launch.json').write_text(json.dumps(metadata, indent=2) + '\n')
    host_status(paths, 'running', **metadata, pid=os.getpid())
    try:
        with (paths.root / 'controller.log').open('wb') as log:
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, start_new_session=True,
                                       stdout=log, stderr=subprocess.STDOUT)
            host_status(paths, 'running', **metadata, pid=os.getpid(), launcher_child_pid=process.pid)
            returncode = process.wait()
    except KeyboardInterrupt:
        if args.runtime == 'docker':
            subprocess.run(['docker', 'rm', '--force', f'codex-{paths.root.name}'],
                           env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        if 'process' in locals():
            try:
                os.killpg(process.pid, signal.SIGINT)
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            except ProcessLookupError:
                pass
        host_status(paths, 'interrupted', **metadata)
        return 130
    except OSError:
        host_status(paths, 'launch_failed', **metadata)
        raise
    if returncode or args.smoke_test:
        host_status(paths, 'failed' if returncode else 'smoke_complete', returncode=returncode, **metadata)
        return returncode
    try:
        snapshot = frozen_submission(paths)
    except (OSError, ValueError, RuntimeError):
        host_status(paths, 'submission_rejected', **metadata)
        raise
    print(f'Frozen submission: {snapshot}', flush=True)
    if not args.no_grade:
        grader_env = os.environ.copy()
        grader_env['PAPERBENCH_ROOT'] = str(args.paperbench_root.resolve())
        host_status(paths, 'grading', **metadata)
        try:
            grade = subprocess.run(['bash', str(PROJECT_ROOT / 'scripts/evaluate_submission.sh'),
                                   args.paper_id, str(snapshot), str(paths.root / 'evaluation')],
                                  cwd=PROJECT_ROOT, env=grader_env, check=False).returncode
        except KeyboardInterrupt:
            host_status(paths, 'grading_interrupted', **metadata)
            return 130
        except OSError:
            host_status(paths, 'grading_failed', **metadata)
            raise
        host_status(paths, 'complete' if grade == 0 else 'grading_failed', returncode=grade, **metadata)
        return grade
    host_status(paths, 'complete', graded=False, **metadata)
    return 0


def main(argv=None, *, docker_default=True) -> int:
    args = parser(docker_default=docker_default).parse_args(argv)
    try:
        return launch(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'Launch failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
