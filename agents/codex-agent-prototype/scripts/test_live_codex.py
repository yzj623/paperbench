#!/usr/bin/env python3
"""Bounded authenticated smoke test. Synthetic inputs only; never run grading."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import secrets
import sys
from unittest.mock import patch

sys.path.insert(0, '/opt/codex-config')
import controller
from app_server import AppServer, AppServerError
from protocol.storage import atomic_json_write
from tool_schemas import function


def prefix(text):
    return [{'type': 'message', 'role': 'user', 'content': [
        {'type': 'input_text', 'text': 'Current active marker: ' + text}]}]


async def probe(model, root):
    markers = ['prefix-A-' + secrets.token_hex(4), 'prefix-B-' + secrets.token_hex(4)]
    seen = []
    async with asyncio.timeout(120):
        async with AppServer(cwd=root, codex_home=Path('/run-auth')) as server:
            try:
                account = (await server.request('account/read', {'refreshToken': False})).get('account') or {}
            except Exception:
                print('Sanitized app-server diagnostic:', server.stderr_tail, flush=True)
                raise
            assert account.get('type') == 'chatgpt', 'ChatGPT authentication unavailable'
            print('Authentication:', account.get('type'), 'plan:', account.get('planType'), flush=True)
            server.trace_path = root / 'prefix-probe.jsonl'
            models = []
            cursor = None
            while True:
                page = await server.request('model/list', {'cursor': cursor, 'limit': 100})
                models.extend(page.get('data', []))
                cursor = page.get('nextCursor')
                if not cursor: break
            catalog_match = any(item.get('model') == model or item.get('id') == model for item in models)
            print('Requested model listed in catalog:', catalog_match, '; checking actual inference without substitution', flush=True)
            tool = function('report_context', 'Report the marker from the current active prefix. Do not infer it from older history.',
                            {'marker': {'type': 'string'}}, ['marker'])
            result = await server.request('thread/start', {
                'model': model, 'cwd': str(root), 'approvalPolicy': 'never', 'sandbox': 'read-only', 'ephemeral': True,
                'config': {'ppblab_dynamic_tools_only': True, 'web_search': 'disabled',
                           'skills': {'include_instructions': False, 'bundled': {'enabled': False}}},
                'baseInstructions': 'This is a tiny protocol test. Call report_context with the current active marker. '
                    'If its result requests another call, read the latest prefix and call it again. After done, reply PASS. '
                    'Do not call tools concurrently. Do not perform any other task.',
                'dynamicTools': [tool]})
            assert result.get('model') == model, 'Server resolved a different model'
            thread = result['thread']['id']
            await server.set_prefix(thread, prefix(markers[0]))
            turn = (await server.request('turn/start', {'threadId': thread,
                'input': [{'type': 'text', 'text': 'Start the protocol test by reporting the current active marker.'}]}))['turn']['id']
            while True:
                event = await server.next_event()
                method, params = event.get('method'), event.get('params', {})
                if 'id' in event:
                    assert method == 'item/tool/call', 'Unexpected tool/server request'
                    assert params.get('tool') == 'report_context' and len(seen) < 2, 'Unexpected tool call'
                    assert params['arguments'].get('marker') == markers[len(seen)], 'Model did not see the current prefix'
                    seen.append(params['arguments']['marker'])
                    if len(seen) == 1:
                        await server.set_prefix(thread, prefix(markers[1]))
                        await server.respond(event['id'], 'The active prefix has changed. Call report_context once more using the current prefix, not the older marker.')
                    else:
                        await server.respond(event['id'], 'done; reply PASS and stop calling tools')
                elif method == 'error' and not params.get('willRetry', False):
                    raise AppServerError(str(params.get('error', {})))
                elif method == 'turn/completed' and params.get('turn', {}).get('id') == turn:
                    assert params['turn']['status'] == 'completed', params['turn'].get('error')
                    break
            assert seen == markers
            await server.set_prefix(thread, [])
    print('Real inference, dynamic tools, in-turn prefix replacement and clear: OK', flush=True)
    return {'auth': 'chatgpt', 'model': model, 'prefix_markers': seen, 'prefix_test': 'passed'}


async def pipeline(model, root):
    workspace, skills, output = root / 'home', root / 'skills', root / 'pipeline'
    for directory in (workspace / 'paper', workspace / 'submission', skills):
        directory.mkdir(parents=True, exist_ok=True)
    marker = 'SKILL_READ_' + secrets.token_hex(5)
    controller.atomic_text(workspace / 'paper/paper.md',
        'Synthetic protocol fixture, not a research paper. Deliver README.md with the marker from '
        'the locally discovered live-protocol-skill, plus reproduce.sh containing only #!/bin/sh and true. '
        'Do not run commands, install packages, train models or call any external service.')
    controller.atomic_text(workspace / 'paper/addendum.md', 'No additional requirements.')
    controller.atomic_text(workspace / 'instructions.txt', 'Only create the two files described in paper.md. No execution.')
    controller.atomic_text(skills / 'SKILL.md', '---\nname: live-protocol-skill\ndescription: Supplies the mandatory README marker for this synthetic protocol test.\n---\n'
        'Write this exact marker in README.md: ' + marker + '\n')
    args = controller.parser().parse_args(['--paper-id', 'live-protocol-fixture', '--workspace', str(workspace),
        '--output', str(output), '--skills', str(skills), '--codex-home', '/run-auth', '--model', model,
        '--plan-input', str(root / 'intentionally-absent-plan.json'),
        '--planner-timeout', '90', '--checker-timeout', '90', '--main-timeout', '180'])
    planner_prompt = 'Return only a JSON plan with exactly one task, id task_001, name Synthetic deliverables, '
    planner_prompt += 'description matching the supplied public paper. Shape: {"subtasks":[{"id":"task_001","name":"...","description":"..."}]}. '
    planner_prompt += 'No other fields. This is a small protocol test, not a real paper experiment.'
    main_prompt = 'This is a small protocol test. Follow the current visible plan. Read the paper and the '
    main_prompt += 'natively indexed live-protocol-skill with the read tool, then use write_file to create ONLY '
    main_prompt += 'README.md and reproduce.sh under the submission directory as required. Use those bare file names as write_file paths. Do not run any shell commands. '
    main_prompt += 'Call submit_subtask(subtask_id="task_001", reproduction_description pointing to the two files). '
    main_prompt += 'Fix if rejected. When accepted, call final_submit. Tools are sequential, never parallel.'
    checker_prompt = 'Read the public paper and inspect both submitted files using read_file. '
    checker_prompt += 'Require a README marker beginning SKILL_READ_ and reproduce.sh containing only #!/bin/sh and true. '
    checker_prompt += 'Then call submit_review once, with accepted true and issues [] if correct, otherwise accepted false and concrete issues. '
    checker_prompt += 'Never execute code. This is a synthetic protocol test.'
    with patch.object(controller, 'PLANNING_PROMPT', planner_prompt), \
         patch.object(controller, 'EXECUTION_PROMPT', main_prompt), \
         patch.object(controller, 'CHECKER_PROMPT', checker_prompt):
        assert await controller.run(args) == 0, 'Controller pipeline failed; see status.json'
    status = json.loads((output / 'status.json').read_text())
    assert status['accepted'] and status['statuses'] == {'task_001': 'completed'}
    assert marker in (workspace / 'submission/README.md').read_text(), 'Native skill content not applied'
    assert (output / 'frozen/submission/reproduce.sh').is_file(), 'Frozen submission missing'
    print('Planner -> main -> checker -> task acceptance -> final_submit -> frozen submission: OK', flush=True)
    print('Native skill discovery/read: OK', flush=True)
    return {'pipeline': 'passed', 'native_skill': 'passed', 'status': status,
            'note': 'Synthetic prompts only; production prompts and real paper performance not tested.'}


async def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--model', default='gpt-6-luna')
    cli.add_argument('--output', type=Path, default=Path('/test-output'))
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {'model': args.model}
    try:
        summary.update(await probe(args.model, args.output))
        summary.update(await pipeline(args.model, args.output))
        summary['ok'] = True
    except Exception as exc:
        summary.update(ok=False, error=controller.redact_for_log(f'{type(exc).__name__}: {exc}'))
        raise
    finally:
        atomic_json_write(args.output / 'summary.json', summary)


if __name__ == '__main__':
    asyncio.run(main())
