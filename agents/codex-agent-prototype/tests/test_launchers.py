"""Launcher contracts with temporary local fixtures; never invoke real Docker."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import docker_runtime
import run_paperbench as launcher
import runtime_config as config


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.args = config.parser().parse_args(['--paper-id', 'example', '--runs-dir', str(self.root / 'runs'),
            '--auth-dir', str(self.root / 'auth'), '--skills-root', str(self.root / 'skills'),
            '--paperbench-root', str(self.root / 'official')])
        self.paths = config.RunPaths(self.root / 'run')

    def ready(self):
        self.args.auth_dir.mkdir()
        (self.args.skills_root / 'example/skill').mkdir(parents=True)
        paper = self.args.paperbench_root / 'project/paperbench/data/papers/example'
        paper.mkdir(parents=True)
        for name in ('paper.md', 'addendum.md', 'blacklist.txt', 'rubric.json', 'config.yaml'):
            (paper / name).write_text(name)
        instructions = self.args.paperbench_root / 'project/paperbench/paperbench/instructions/code_only_instructions.txt'
        instructions.parent.mkdir(parents=True)
        instructions.write_text('public instructions')

    def init_run(self):
        self.paths.root.mkdir()
        launcher.prepare_workspace(self.args, self.paths)

    def test_defaults_and_optional_model(self):
        self.assertEqual(self.args.runtime, 'docker')
        self.assertIsNone(self.args.planner_timeout)
        self.assertIsNone(self.args.checker_timeout)
        self.assertTrue(self.args.checker)
        self.assertIsNone(self.args.main_timeout)
        self.assertFalse(self.args.strip_irrelevant_developer_context)
        command = config.controller_command(self.args, self.paths, docker=True)
        self.assertNotIn('--no-checker', command)
        self.assertNotIn('--strip-irrelevant-developer-context', command)
        self.assertEqual(command[command.index('--model') + 1], 'gpt-6-luna')
        self.assertNotIn('--main-timeout', command)
        self.assertIn('/opt/codex-config/controller.py', command)
        self.assertIn('CODEX_APP_SERVER_COMMAND=["/opt/codex/bin/codex-app-server"]',
                      docker_runtime.docker_command(self.args, self.paths))

    def test_paperbench_default_is_project_local_with_explicit_override(self):
        defaults = config.parser().parse_args(['--paper-id', 'example'])
        self.assertEqual(defaults.paperbench_root, config.PROJECT_ROOT / 'vendor/paperbench-official')
        self.assertTrue(defaults.paperbench_root.is_relative_to(config.PROJECT_ROOT))
        self.assertEqual(self.args.paperbench_root, self.root / 'official')
        self.assertIn('vendor/', (config.PROJECT_ROOT / '.dockerignore').read_text().splitlines())

    def test_generated_argv_parses_with_real_controller_parser(self):
        import controller

        for docker in (True, False):
            for smoke in (True, False):
                with self.subTest(docker=docker, smoke=smoke):
                    self.args.smoke_test = smoke
                    command = config.controller_command(self.args, self.paths, docker=docker)
                    parsed = controller.parser().parse_args(command[2:])
                    self.assertEqual(parsed.paper_id, 'example')
                    self.assertIsNone(parsed.planner_timeout)
                    self.assertIsNone(parsed.checker_timeout)
                    self.assertIsNone(parsed.main_timeout)
                    self.assertEqual(parsed.model, 'gpt-6-luna')
                    self.assertEqual(parsed.smoke_test, smoke)
                    self.assertTrue(parsed.checker)
                    self.assertFalse(parsed.strip_irrelevant_developer_context)
                    self.assertEqual(parsed.workspace, Path('/home') if docker else self.paths.workspace)
                    self.assertEqual(parsed.output, Path('/run-output') if docker else self.paths.output)
                    self.assertEqual(parsed.plan_input, Path('/run-input/plan.json') if docker else self.paths.inputs / 'plan.json')
                    self.assertEqual(parsed.skills, Path('/skills') if docker else
                                     self.paths.inputs / 'smoke-skills' if smoke else self.args.skills_root / 'example/skill')
                    self.assertEqual(parsed.codex_home, Path('/run-output/codex-home') if docker and smoke else
                                     Path('/run-auth') if docker else self.paths.root / 'codex-home' if smoke else self.args.auth_dir)
        self.args.model = 'test-model-from-user'
        self.args.main_timeout = 120
        self.args.checker = False
        self.args.strip_irrelevant_developer_context = True
        command = config.controller_command(self.args, self.paths, docker=True)
        self.assertIn('--no-checker', command)
        self.assertIn('--strip-irrelevant-developer-context', command)
        parsed = controller.parser().parse_args(command[2:])
        self.assertEqual(parsed.model, 'test-model-from-user')
        self.assertEqual(parsed.main_timeout, 120)
        self.assertFalse(parsed.checker)
        self.assertTrue(parsed.strip_irrelevant_developer_context)

    def test_one_container_mounts_and_unlimited_resources(self):
        command = docker_runtime.docker_command(self.args, self.paths)
        text = ' '.join(command)
        self.assertEqual(command[:2], ['docker', 'run'])
        self.assertIn('0:0', command)
        self.assertIn(f'src={self.paths.workspace},dst=/home', text)
        self.assertNotIn(f'src={self.paths.workspace},dst={self.paths.workspace}', text)
        self.assertNotIn('/var/run/docker.sock', text)
        self.assertIn(f'src={self.args.skills_root}/example/skill,dst=/skills,readonly', text)
        self.assertIn(f'src={self.args.auth_dir},dst=/run-auth', text)
        for flag in ('--cpus', '--memory', '--gpus', '--read-only'):
            self.assertNotIn(flag, command)

    def test_optional_limits_proxy_and_fixed_plan(self):
        self.args.cpus, self.args.memory, self.args.gpus = 2, '4g', 'all'
        self.args.proxy = 'http://127.0.0.1:7890'
        self.args.plan = self.root / 'external-plan.json'
        command = docker_runtime.docker_command(self.args, self.paths)
        self.assertIn('--cpus', command)
        self.assertIn('--memory', command)
        self.assertEqual(command[command.index('--memory-swap') + 1], '4g')
        self.assertIn('--gpus', command)
        self.assertIn('HTTP_PROXY=http://host.docker.internal:7890', command)
        self.assertIn('/run-input/plan.json', command)

    def test_auth_free_smoke_mount(self):
        self.args.smoke_test = True
        command = docker_runtime.docker_command(self.args, self.paths)
        self.assertNotIn('dst=/run-auth', ' '.join(command))
        self.assertIn('CODEX_HOME=/run-output/codex-home', command)
        self.assertIn(f'src={self.paths.inputs}/smoke-skills,dst=/skills,readonly', ' '.join(command))

    def test_dry_run_no_auth_no_writes_no_subprocess(self):
        self.args.dry_run = True
        with patch.object(launcher.subprocess, 'Popen') as process, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(launcher.launch(self.args), 0)
        process.assert_not_called()
        self.assertFalse(self.args.runs_dir.exists())

    def test_auth_validation_never_reads_contents(self):
        self.ready()
        with patch.object(Path, 'read_text', side_effect=AssertionError('must not read auth')):
            env = config.agent_environment(self.args, self.paths)
            self.assertEqual(env['CODEX_HOME'], str(self.args.auth_dir))
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'secret', 'USTC_LLM_API_KEY': 'secret', 'CODEX_HOME': '/personal'}):
            env = config.agent_environment(self.args, self.paths)
            self.assertNotIn('OPENAI_API_KEY', env)
            self.assertNotIn('USTC_LLM_API_KEY', env)
            self.assertNotEqual(env['CODEX_HOME'], '/personal')

    def test_real_controller_roles_share_explicit_auth_without_reading(self):
        import controller

        parsed = controller.parser().parse_args(config.controller_command(self.args, self.paths, docker=False)[2:])
        with patch.object(Path, 'read_text', side_effect=AssertionError('must not read credentials')), \
             patch.object(Path, 'read_bytes', side_effect=AssertionError('must not read credentials')):
            for role in ('planner', 'checker', 'main'):
                with self.subTest(role=role):
                    environment, auth_root = controller.role_environment(parsed, role)
                    self.assertEqual(auth_root, self.args.auth_dir)
                    self.assertEqual(environment['CODEX_HOME'], str(self.args.auth_dir))

    def test_prepare_public_allowlist_git_agent_env(self):
        self.ready()
        self.init_run()
        self.assertTrue((self.paths.workspace / 'paper/blacklist.txt').is_file())
        self.assertFalse((self.paths.workspace / 'paper/rubric.json').exists())
        self.assertFalse((self.paths.workspace / 'paper/config.yaml').exists())
        self.assertTrue((self.paths.workspace / 'submission/.git').is_dir())
        self.assertTrue((self.paths.workspace / 'submission/.git/hooks/pre-commit').is_file())
        self.assertNotIn('WORKSPACE_HOST_PATH=', (self.paths.workspace / 'agent.env').read_text())

    def test_prepare_copies_only_public_solver_assets_and_env_example(self):
        self.ready()
        solvers = self.args.paperbench_root / 'project/paperbench/paperbench/solvers'
        solvers.mkdir()
        (solvers / 'agent.env.example').write_text('# Public environment template\n')
        (solvers / 'pre-commit').write_bytes(b'#!/bin/bash\r\nexit 0\r\n')
        (solvers / 'launch.json').write_text('{"version": "0.2.0"}\n')
        (solvers / 'agent.env').write_text('DO_NOT_COPY_PRIVATE_ENV=fixture\n')
        self.init_run()
        self.assertEqual((self.paths.workspace / 'agent.env').read_text(), '# Public environment template\n')
        self.assertEqual((self.paths.workspace / 'submission/.git/hooks/pre-commit').read_bytes(), b'#!/bin/bash\nexit 0\n')
        self.assertEqual((self.paths.workspace / '.vscode/launch.json').read_text(), '{"version": "0.2.0"}\n')

    def test_smoke_has_offline_inputs_without_official(self):
        self.args.smoke_test = True
        self.init_run()
        self.assertTrue((self.paths.workspace / 'paper/paper.md').is_file())
        self.assertTrue((self.paths.workspace / 'instructions.txt').is_file())
        self.assertTrue((self.paths.inputs / 'smoke-skills/offline-smoke/SKILL.md').is_file())

    def test_actual_host_smoke_controller_contract_no_auth(self):
        self.args.runtime = 'host'
        self.args.smoke_test = True
        self.args.plan = self.root / 'fixed-plan.json'
        self.args.plan.write_text(json.dumps({'subtasks': [{
            'id': 'task_001', 'name': 'Offline validation', 'description': 'Validate fixtures without inference.'}]}))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(launcher.launch(self.args), 0)
        run = next(self.args.runs_dir.iterdir())
        report = json.loads((run / 'controller.log').read_text())
        self.assertTrue(report['ok'])
        self.assertEqual(report['fixed_plan_tasks'], 1)
        self.assertEqual(report['skills'], 1)
        self.assertFalse(self.args.auth_dir.exists())
        self.assertEqual(json.loads((run / 'host_status.json').read_text())['phase'], 'smoke_complete')

    def complete_snapshot(self):
        self.paths.output.mkdir(parents=True)
        (self.paths.output / 'protocol').mkdir()
        (self.paths.output / 'protocol/tool_calls.jsonl').write_text(json.dumps({
            'event': 'tool_result', 'tool': 'final_submit', 'result': {'ok': True}}) + '\n')
        (self.paths.output / 'status.json').write_text(json.dumps({'phase': 'complete', 'accepted': True}))
        snapshot = self.paths.output / 'frozen/submission'
        snapshot.mkdir(parents=True)
        (snapshot / 'README.md').write_text('accepted version')
        return snapshot

    def test_frozen_submission_uses_controller_snapshot(self):
        snapshot = self.complete_snapshot()
        (self.paths.workspace / 'submission').mkdir(parents=True)
        (self.paths.workspace / 'submission/README.md').write_text('changed live version')
        self.assertEqual(launcher.frozen_submission(self.paths), snapshot)
        self.assertEqual((snapshot / 'README.md').read_text(), 'accepted version')

    def test_zero_exit_alone_and_rejected_status_do_not_grade(self):
        self.complete_snapshot()
        (self.paths.output / 'status.json').write_text(json.dumps({'phase': 'complete', 'accepted': False}))
        with self.assertRaises(RuntimeError):
            launcher.frozen_submission(self.paths)

        (self.paths.output / 'status.json').write_text(json.dumps({'phase': 'complete', 'accepted': True}))
        (self.paths.output / 'protocol/tool_calls.jsonl').write_text('')
        with self.assertRaises(RuntimeError):
            launcher.frozen_submission(self.paths)

    def test_grade_calls_local_script_only_with_frozen_snapshot(self):
        self.ready()
        child = Mock(pid=12345)
        child.wait.return_value = 0
        def prepare(args, paths):
            paths.output.mkdir()
            (paths.output / 'protocol').mkdir()
            (paths.output / 'protocol/tool_calls.jsonl').write_text(json.dumps({
                'event': 'tool_result', 'tool': 'final_submit', 'result': {'ok': True}}) + '\n')
            (paths.output / 'status.json').write_text(json.dumps({'phase': 'complete', 'accepted': True}))
            (paths.output / 'frozen/submission').mkdir(parents=True)
        with patch.object(launcher, 'PROJECT_ROOT', self.root), patch.object(Path, 'is_file', return_value=True), \
             patch.object(launcher, 'prepare_workspace', side_effect=prepare), \
             patch.object(launcher.subprocess, 'Popen', return_value=child), \
             patch.object(launcher.subprocess, 'run', return_value=Mock(returncode=0)) as grader, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(launcher.launch(self.args), 0)
        run = next(self.args.runs_dir.iterdir())
        self.assertEqual(grader.call_args.args[0], ['bash', str(self.root / 'scripts/evaluate_submission.sh'),
                                                  'example', str(run / 'output/frozen/submission'), str(run / 'evaluation')])
        self.assertEqual(json.loads((run / 'host_status.json').read_text())['phase'], 'complete')

    def test_unique_runs_and_smoke_skip_grade(self):
        self.args.smoke_test = True
        child = Mock(pid=12345)
        child.wait.return_value = 0
        with patch.object(launcher, 'PROJECT_ROOT', self.root), patch.object(Path, 'is_file', return_value=True), \
             patch.object(launcher.subprocess, 'Popen', return_value=child), \
             patch.object(launcher, 'prepare_workspace'), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(launcher.launch(self.args), 0)
            self.assertEqual(launcher.launch(self.args), 0)
        runs = list(self.args.runs_dir.iterdir())
        self.assertEqual(len(runs), 2)
        for run in runs:
            self.assertEqual(json.loads((run / 'host_status.json').read_text())['phase'], 'smoke_complete')
            self.assertTrue((run / 'launch.json').is_file())
            self.assertFalse((run / 'evaluation').exists())

    def test_interrupt_removes_exact_container(self):
        self.args.smoke_test = True
        child = Mock(pid=12345)
        child.wait.side_effect = [KeyboardInterrupt, 0]
        # Workspace git initialization uses the real local git; Docker stays mocked.
        real_run = subprocess.run
        docker_calls = []
        def local_only(command, **kwargs):
            if command[0] == 'docker':
                docker_calls.append(command)
                return Mock(returncode=0)
            return real_run(command, **kwargs)
        with patch.object(launcher, 'PROJECT_ROOT', self.root), patch.object(Path, 'is_file', return_value=True), \
             patch.object(launcher.subprocess, 'Popen', return_value=child), \
             patch.object(launcher.subprocess, 'run', side_effect=local_only), \
             patch.object(launcher.os, 'killpg') as killpg, contextlib.redirect_stdout(io.StringIO()):
            # Mock setup avoids subprocess.run's internal Popen while testing interruption.
            with patch.object(launcher, 'prepare_workspace'):
                self.assertEqual(launcher.launch(self.args), 130)
        run = next(self.args.runs_dir.iterdir())
        self.assertEqual(docker_calls, [['docker', 'rm', '--force', f'codex-{run.name}']])
        killpg.assert_called_once()
        self.assertEqual(json.loads((run / 'host_status.json').read_text())['phase'], 'interrupted')

    def test_reject_path_traversal_and_credential_proxy(self):
        with self.assertRaises(ValueError):
            config.validate_paper_id('../outside')
        with self.assertRaises(ValueError):
            docker_runtime.docker_proxy('http://user:secret@localhost:7890')

    def test_build_recipes_are_locked_and_do_not_install_rust(self):
        dockerfile = (config.PROJECT_ROOT / 'Dockerfile.codex').read_text()
        self.assertIn('FROM rust:1.95.0-bookworm', dockerfile)
        self.assertIn('FROM pb-env:latest', dockerfile)
        self.assertIn('ARG BUILD_JOBS=1', dockerfile)
        self.assertIn('--locked -p codex-app-server --bin codex-app-server', dockerfile)
        for name in ('build_codex.sh', 'build_codex_image.sh'):
            result = subprocess.run(['bash', str(config.PROJECT_ROOT / name), '--dry-run'], capture_output=True, text=True, check=True)
            self.assertIn('1', result.stdout)
        script = (config.PROJECT_ROOT / 'build_codex.sh').read_text()
        self.assertNotIn('rustup toolchain install', script)
        self.assertIn('.build-tools/cargo/bin/rustup', script)


if __name__ == '__main__':
    unittest.main()
