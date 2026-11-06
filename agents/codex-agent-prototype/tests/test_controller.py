import asyncio
import json
import os
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from controller import (parser, run, run_role, role_config, role_environment,
                        publish_agent_view, LocalRuntime, LocalShellComputer,
                        MainDispatcher, LocalSnapshotter)
from app_server import AppServerError
from tool_schemas import role_tools, validate_call
from protocol.prompts import render_plan_message
from protocol.storage import PlanStore
from protocol.readonly import SafeWorkspace, ReadOnlyToolDispatcher, PathAccessError
from protocol.state_machine import ResearchAgentTools, ResearchToolError
from protocol.async_jobs import ManagedJobManager
from test_app_server import FAKE_SERVER


PLAN = {'subtasks': [{'id': 'task_001', 'name': 'Implementation',
                     'description': 'Implement paper and final packaging.'}]}


class ControllerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        workspace = self.root / 'home'
        (workspace / 'paper').mkdir(parents=True)
        (workspace / 'submission').mkdir()
        for relative in ('paper/paper.md', 'paper/addendum.md', 'instructions.txt'):
            (workspace / relative).write_text('Public input\n')
        skills = self.root / 'skills'
        skills.mkdir()
        (skills / 'SKILL.md').write_text('---\nname: local\ndescription: Local skill\n---\n')
        self.args = parser().parse_args(['--paper-id', 'fixture', '--workspace', str(workspace),
            '--output-dir', str(self.root / 'output'), '--skills-root', str(skills),
            '--codex-home', str(self.root / 'auth'), '--plan', str(self.root / 'plan.json')])
        self.args.app_server = [sys.executable, '-u', '-c', FAKE_SERVER]
        self.args.model = 'fixture-model'
        self.args.main_timeout = 10
        self.args.planner_timeout = 5
        self.args.checker_timeout = 5

    async def asyncTearDown(self):
        self.temporary.cleanup()

    async def test_full_jsonl_plan_repair_main_followup_nested_checker_final(self):
        with patch.dict(os.environ, {'FAKE_MODE': 'normal'}):
            self.assertEqual(await run(self.args), 0)
        status = json.loads((self.args.output / 'status.json').read_text())
        self.assertEqual(status['phase'], 'complete')
        self.assertTrue(status['accepted'])
        self.assertEqual(status['statuses'], {'task_001': 'completed'})
        self.assertTrue((self.args.output / 'submissions/submission.tar.gz').is_file())
        self.assertEqual((self.args.workspace / '.agents/skills').resolve(), self.args.skills)
        for path in [self.args.output / 'planner.jsonl', self.args.output / 'main.jsonl',
                     self.args.output / 'checker/task_001-review-1.jsonl']:
            events = [json.loads(line) for line in path.read_text().splitlines()]
            thread = next(e['params'] for e in events if e.get('method') == 'fixture/thread')
            self.assertEqual(thread['home'], str(self.args.codex_home))
            self.assertEqual(thread['model'], 'fixture-model')
            self.assertEqual(sum(e.get('method') == 'fixture/thread' for e in events), 1)
        events = [json.loads(line) for line in (self.args.output / 'main.jsonl').read_text().splitlines()]
        self.assertTrue(any(e.get('direction') == 'client-tool-response' and
                            e['id'] == 'tool-final_submit' for e in events))
        self.assertTrue(any(e.get('method') == 'item/completed' and
                            e.get('params', {}).get('item', {}).get('type') == 'commandExecution'
                            for e in events))

    async def test_no_checker_end_to_end_skips_checker_process(self):
        self.args.plan_input.write_text(json.dumps(PLAN))
        self.args.checker = False
        with patch.dict(os.environ, {'FAKE_MODE': 'normal'}):
            self.assertEqual(await run(self.args), 0)
        status = json.loads((self.args.output / 'status.json').read_text())
        self.assertEqual(status['phase'], 'complete')
        self.assertTrue(status['accepted'])
        self.assertFalse((self.args.output / 'checker').exists())

    async def test_failed_provider_stops_without_followup(self):
        self.args.plan_input.write_text(json.dumps(PLAN))
        with patch.dict(os.environ, {'FAKE_MODE': 'provider-error'}):
            self.assertEqual(await run(self.args), 1)
        self.assertEqual(json.loads((self.args.output / 'status.json').read_text())['phase'], 'failed')

    async def test_prefix_update_error_prevents_main_sampling(self):
        self.args.plan_input.write_text(json.dumps(PLAN))
        with patch.dict(os.environ, {'FAKE_MODE': 'prefix-error'}):
            self.assertEqual(await run(self.args), 1)
        events = [json.loads(line) for line in (self.args.output / 'main.jsonl').read_text().splitlines()]
        self.assertFalse(any(e.get('method') == 'turn/start' for e in events))
        self.assertIn('invalid prefix', (self.args.output / 'status.json').read_text())

    async def test_failed_provider_no_extra_turns(self):
        self.args.plan_input.write_text(json.dumps(PLAN))
        with patch.dict(os.environ, {'FAKE_MODE': 'provider-error'}):
            self.assertEqual(await run(self.args), 1)
        status = json.loads((self.args.output / 'status.json').read_text())
        self.assertEqual(status['phase'], 'failed')
        self.assertIn('provider exhausted retries', status['error'])
        text = (self.args.output / 'main.jsonl').read_text()
        self.assertEqual(text.count('"status": "failed"'), 1)

    async def test_retrying_provider_notification_does_not_abort(self):
        self.args.plan_input.write_text(json.dumps(PLAN))
        with patch.dict(os.environ, {'FAKE_MODE': 'retry-then-success'}):
            self.assertEqual(await run(self.args), 0)
        self.assertTrue(json.loads((self.args.output / 'status.json').read_text())['accepted'])

    async def test_three_fresh_checker_failures_return_tool_error_to_main(self):
        self.args.plan_input.write_text(json.dumps(PLAN))
        with patch.dict(os.environ, {'FAKE_MODE': 'checker-unavailable'}):
            self.assertEqual(await run(self.args), 1)
        status = json.loads((self.args.output / 'status.json').read_text())
        self.assertEqual(status['statuses'], {'task_001': 'uncompleted'})
        self.assertIn('main provider subsequently failed', status['error'])
        self.assertEqual(len(list((self.args.output / 'checker').glob('*.jsonl'))), 3)
        main_events = (self.args.output / 'main.jsonl').read_text()
        self.assertIn('fixture/checkerErrorReturned', main_events)
        self.assertEqual(PlanStore(self.args.output / 'protocol').failure_count('task_001'), 0)

    async def test_native_approval_request_is_declined_without_stopping_protocol(self):
        self.args.plan_input.write_text(json.dumps(PLAN))
        with patch.dict(os.environ, {'FAKE_MODE': 'native-request'}):
            self.assertEqual(await run(self.args), 0)
        self.assertIn('"method": "item/commandExecution/requestApproval"',
                      (self.args.output / 'main.jsonl').read_text())

    def test_published_prefix_uses_original_renderer(self):
        store = PlanStore(self.root / 'protocol')
        store.initialize({'subtasks': PLAN['subtasks'] + [
            {'id': 'task_002', 'name': 'Hidden', 'description': 'Later work'}]})
        self.assertEqual(len(store.render()), 2)
        publish_agent_view(store, self.args.output)
        actual = (self.args.output / 'agent_view.json').read_text()
        self.assertEqual(actual, render_plan_message(store.render())['content'])
        self.assertNotIn('task_002', actual)
        store.set_status('task_001', 'completed', reason='accepted')
        publish_agent_view(store, self.args.output)
        self.assertIn('task_002', (self.args.output / 'agent_view.json').read_text())

    def test_role_scope_shared_auth_native_router_and_workload_marker(self):
        for role in ('planner', 'checker', 'main'):
            runtime = LocalRuntime()
            environment, home = role_environment(self.args, role, runtime)
            self.assertEqual(home, self.args.codex_home)
            self.assertEqual(environment['CODEX_HOME'], str(home))
            self.assertNotIn('PI_RUN_AGENT_TOKEN', environment)
            if role == 'main':
                self.assertEqual(environment['PAPERBENCH_PROCESS_TOKEN'], runtime.marker)
            else:
                self.assertNotIn('PAPERBENCH_PROCESS_TOKEN', environment)
            config = role_config(self.args, role)
            self.assertFalse(config['ppblab_dynamic_tools_only'])
            self.assertNotIn('tools', config)
            self.assertNotIn('agents', config)
            self.assertNotIn('features', config)
            self.assertNotIn('model_context_window', config)
            self.assertNotIn('ppblab_plan_path', config)
            self.assertNotIn('ppblab_plan_snapshot_dir', config)
            if role != 'main':
                self.assertFalse(config['skills']['include_instructions'])
                self.assertFalse(config['skills']['bundled']['enabled'])
                self.assertEqual(config['skills']['config'], [{'path': str(self.args.skills / 'SKILL.md'), 'enabled': False}])
                self.assertNotIn('ppblab_plan_path', config)
                with self.assertRaises(ValueError):
                    validate_call(role_tools(role), 'bash', {'cmd': 'true'})
            else:
                self.assertEqual(config['skills'], {'bundled': {'enabled': False}})

    def test_strip_irrelevant_developer_context_config_is_opt_in(self):
        for role in ('planner', 'checker', 'main'):
            config = role_config(self.args, role)
            self.assertNotIn('agents', config)
            self.assertNotIn('features', config)
        self.args.strip_irrelevant_developer_context = True
        for role in ('planner', 'checker', 'main'):
            config = role_config(self.args, role)
            self.assertEqual(config['agents'], {'enabled': False})
            self.assertFalse(config['include_collaboration_mode_instructions'])
            self.assertFalse(config['include_apps_instructions'])
            self.assertEqual(config['features'],
                             {'apps': False, 'plugins': False, 'tool_suggest': False})
            self.assertNotIn('include_permissions_instructions', config)
            self.assertNotIn('include_environment_context', config)
            if role == 'main':
                self.assertEqual(config['skills'], {'bundled': {'enabled': False}})
            else:
                self.assertFalse(config['skills']['include_instructions'])

    async def test_native_command_descendant_is_detected_by_submission_gate(self):
        runtime = LocalRuntime()
        environment, _ = role_environment(self.args, 'main', runtime)
        process = subprocess.Popen(['sleep', '5'], env=environment)
        try:
            await asyncio.sleep(0.05)
            self.assertIn(process.pid, {item.pid for item in await runtime.active_processes()})
            runtime.main_pid = process.pid
            self.assertNotIn(process.pid, {item.pid for item in await runtime.active_processes()})
        finally:
            process.terminate()
            process.wait(timeout=2)

    async def test_code_mode_host_is_not_an_agent_workload(self):
        proc = self.root / 'proc'
        proc.mkdir()
        binaries = self.root / 'bin'
        binaries.mkdir()
        server_exe = binaries / 'codex-app-server'
        host_exe = binaries / 'codex-code-mode-host'
        worker_exe = binaries / 'sleep'
        for executable in (server_exe, host_exe, worker_exe):
            executable.touch()

        runtime = LocalRuntime(proc_root=proc)
        runtime.main_pid = 100

        def add_process(pid, parent_pid, executable):
            directory = proc / str(pid)
            directory.mkdir()
            (directory / 'exe').symlink_to(executable)
            (directory / 'environ').write_bytes(
                f'PAPERBENCH_PROCESS_TOKEN={runtime.marker}\0'.encode())
            (directory / 'stat').write_text(f'{pid} (process) S {parent_pid} 0 0\n')
            (directory / 'cmdline').write_bytes(str(executable).encode() + b'\0')

        add_process(100, 1, server_exe)
        add_process(110, 100, host_exe)
        add_process(120, 110, worker_exe)
        add_process(130, 100, worker_exe)
        add_process(140, 110, host_exe)
        self.assertEqual({item.pid for item in await runtime.active_processes()},
                         {120, 130, 140})

    def test_checker_readonly_exact_allowlist(self):
        briefing = self.root / 'briefing.txt'
        briefing.write_text('Briefing')
        readonly = ReadOnlyToolDispatcher(SafeWorkspace(public_root=self.args.workspace / 'paper',
            workspace_root=self.args.workspace, checker_instructions_file=briefing))
        for path in (str(self.args.skills / 'SKILL.md'), '/home/instructions.txt', str(self.args.codex_home / 'auth.json')):
            with self.assertRaises(PathAccessError):
                readonly.call('read_file', {'path': path})
        outside = self.root / 'outside.txt'
        outside.write_text('private')
        (self.args.workspace / 'submission/escape').symlink_to(outside)
        with self.assertRaises(PathAccessError):
            readonly.call('read_file', {'path': '/home/submission/escape'})

    async def test_no_checker_submission_is_accepted_directly(self):
        runtime = LocalRuntime()
        store = PlanStore(self.root / 'protocol-no-checker')
        store.initialize(PLAN)
        manager = ResearchAgentTools(store=store, runtime=runtime, checker_factory=None,
            completer_config=None, public_paper_root=self.args.workspace / 'paper',
            workspace_root=self.args.workspace, checker_enabled=False)
        result = await manager.submit_subtask(
            subtask_id='task_001', reproduction_description='Implemented the subtask.')
        self.assertIn('completed', result)
        self.assertEqual(store.status('task_001'), 'completed')
        self.assertFalse((self.args.output / 'checker').exists())

    async def test_workload_detection_and_final_gate(self):
        runtime = LocalRuntime()
        computer = LocalShellComputer(self.args.workspace, runtime)
        store = PlanStore(self.root / 'protocol')
        store.initialize(PLAN)
        store.set_status('task_001', 'completed', reason='accepted')
        for filename in ('README.md', 'reproduce.sh'):
            (self.args.workspace / 'submission' / filename).write_text('content')
        jobs = ManagedJobManager(store=store, computer=computer, workspace_root=self.args.workspace)
        manager = ResearchAgentTools(store=store, runtime=runtime, checker_factory=None,
            completer_config=None, public_paper_root=self.args.workspace / 'paper',
            workspace_root=self.args.workspace, job_manager=jobs,
            snapshotter=LocalSnapshotter(self.args.workspace, self.args.output))
        self.args.output.mkdir()
        command = asyncio.create_task(computer.send_shell_command('sleep 30'))
        try:
            for _ in range(100):
                if await runtime.active_processes():
                    break
                await asyncio.sleep(.01)
            result = await manager.final_submit()
            self.assertFalse(result.accepted)
            self.assertIn('ACTIVE_AGENT_PROCESSES', [e['code'] for e in result.errors])
        finally:
            command.cancel()
            await asyncio.gather(command, return_exceptions=True)
            await jobs.close()
            await runtime.close()

    def test_cli_defaults_and_aliases(self):
        with patch.dict(os.environ, {'CODEX_HOME': '/explicit-auth'}):
            args = parser().parse_args(['--paper-id', 'fixture', '--app-server', '/patched/server'])
        self.assertIsNone(args.checker_timeout)
        self.assertIsNone(args.main_timeout)
        self.assertFalse(args.strip_irrelevant_developer_context)
        self.assertEqual(args.model, 'gpt-6-luna')
        self.assertEqual(args.codex_home, Path('/explicit-auth'))
        self.assertEqual(args.app_server, ['/patched/server'])

    async def test_smoke_does_not_start_server(self):
        self.args.smoke_test = True
        self.args.app_server = ['/does-not-exist']
        self.assertEqual(await run(self.args), 0)
        self.assertFalse(self.args.codex_home.exists())


if __name__ == '__main__':
    unittest.main()
