"""Offline wire fixtures: no Codex credentials or provider are used."""
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app_server import AppServer, AppServerError, server_command


FAKE_SERVER = r'''
import sys,json,os
def emit(value):
    print(json.dumps(value),flush=True)
def reply(msg,value):
    emit({'id':msg['id'],'result':value})
def notification(method,params):
    emit({'method':method,'params':params})
def completed(text=''):
    notification('item/completed',{'threadId':'thread','turnId':turn,'item':{'type':'agentMessage','phase':'final_answer','text':text}})
    notification('turn/completed',{'threadId':'thread','turn':{'id':turn,'status':'completed'}})
def tool(name,args):
    global pending
    pending=name
    emit({'id':'tool-'+name,'method':'item/tool/call','params':{'threadId':'thread','turnId':turn,'callId':'call-'+name,'namespace':None,'tool':name,'arguments':args}})
def native_main_work():
    os.makedirs('submission',exist_ok=True)
    with open('submission/README.md','w') as out: out.write('Complete implementation documentation')
    with open('submission/reproduce.sh','w') as out: out.write('#!/bin/sh\ntrue\n')
    notification('item/started',{'threadId':'thread','turnId':turn,'item':{'type':'commandExecution','command':'true','cwd':os.getcwd()}})
    notification('item/completed',{'threadId':'thread','turnId':turn,'item':{'type':'commandExecution','status':'completed','exitCode':0,'aggregatedOutput':''}})
    notification('item/started',{'threadId':'thread','turnId':turn,'item':{'type':'fileChange','changes':[{'path':'submission/README.md','kind':'add','diff':''}]}})
    notification('item/completed',{'threadId':'thread','turnId':turn,'item':{'type':'fileChange','status':'completed'}})
    tool('submit_subtask',{'subtask_id':'task_001','reproduction_description':'README.md and reproduce.sh'})
mode=os.environ.get('FAKE_MODE','normal')
role=''
count=0
turn=''
pending=''
prefix=[]
prefix_count=0
for line in sys.stdin:
    msg=json.loads(line)
    method=msg.get('method')
    if method=='initialize':
        assert msg['params']['capabilities']['experimentalApi'] is True
        if mode=='rpc-error':
            emit({'id':msg['id'],'error':{'code':-32000,'message':'provider unavailable api_key=private-value','apiKey':'secret'}})
        else: reply(msg,{'serverInfo':{'name':'fixture'}})
    elif method=='initialized': pass
    elif method=='model/list':
        reply(msg,{'data':[{'id':'other','model':'other-model','isDefault':False},{'id':'default','model':'fixture-model','isDefault':True}], 'nextCursor':None})
    elif method=='probe':
        notification('probe/event',{'value':1})
        reply(msg,{'echo':msg['params']})
    elif method=='fail':
        print('provider exhausted retries',file=sys.stderr,flush=True)
        sys.exit(2)
    elif method=='thread/start':
        params=msg['params']
        assert params['config']['ppblab_dynamic_tools_only'] is False
        assert params['approvalPolicy']=='never'
        names={item['name'] for item in params['dynamicTools']}
        role='main' if 'final_submit' in names else 'checker' if 'submit_review' in names else 'planner'
        assert names==({'submit_subtask','skip_subtask','final_submit'} if role=='main' else
                       {'submit_review'} if role=='checker' else set())
        assert params['sandbox']==('danger-full-access' if role=='main' else 'read-only')
        assert params['ephemeral'] is (role!='main')
        if role!='main':
            assert params['config']['skills']['include_instructions'] is False
            assert not any(item['enabled'] for item in params['config']['skills']['config'])
            assert 'ppblab_plan_path' not in params['config']
        reply(msg,{'thread':{'id':'thread'}})
        notification('fixture/thread',{'role':role,'home':os.environ['CODEX_HOME'],'model':params['model'],'config':params['config']})
    elif method=='thread/prefix/set':
        assert msg['params']['threadId']=='thread'
        assert isinstance(msg['params']['messages'],list)
        if mode=='prefix-error':
            emit({'id':msg['id'],'error':{'code':-32602,'message':'invalid prefix'}})
        else:
            prefix=msg['params']['messages']
            prefix_count+=1
            reply(msg,{})
            notification('fixture/prefixSeen',{'messages':prefix,'count':prefix_count})
    elif method=='turn/start':
        if role=='main': assert prefix_count>0
        count+=1
        turn='turn-'+str(count)
        reply(msg,{'turn':{'id':turn,'status':'inProgress'}})
        if mode=='retry-then-success':
            notification('error',{'threadId':'thread','turnId':turn,'willRetry':True,'error':{'message':'transient provider error'}})
        if mode=='provider-error':
            notification('turn/completed',{'threadId':'thread','turn':{'id':turn,'status':'failed','error':{'message':'provider exhausted retries'}}})
        elif mode=='native-request' and role=='main':
            emit({'id':'native','method':'item/commandExecution/requestApproval','params':{'threadId':'thread'}})
        elif role=='planner':
            if count==1: completed('not json')
            else:
                assert 'INVALID_PLAN' in msg['params']['input'][0]['text']
                completed(json.dumps({'subtasks':[{'id':'task_001','name':'Implementation','description':'Implement paper and final packaging.'}]}))
        elif role=='checker':
            if mode=='checker-unavailable':
                notification('turn/completed',{'threadId':'thread','turn':{'id':turn,'status':'failed','error':{'message':'checker provider exhausted retries'}}})
            else: tool('submit_review',{'accepted':True,'summary':'Inspected implementation.','issues':[]})
        elif role=='main':
            if count==1: completed('Ordinary text is not submission.')
            elif mode!='native-request': native_main_work()
    elif 'result' in msg and pending:
        assert msg['result']['contentItems'][0]['type']=='inputText'
        if role=='checker':
            notification('fixture/responseSeen',{'tool':pending})
            completed()
        elif role=='main':
            if pending=='submit_subtask':
                if mode=='checker-unavailable':
                    assert msg['result']['success'] is False
                    assert 'CHECKER_UNAVAILABLE' in msg['result']['contentItems'][0]['text']
                    notification('fixture/checkerErrorReturned',{})
                    notification('turn/completed',{'threadId':'thread','turn':{'id':turn,'status':'failed','error':{'message':'main provider subsequently failed'}}})
                else:
                    assert prefix_count>=2, 'updated prefix must arrive before the review tool result'
                    tool('final_submit',{})
            else:
                notification('fixture/responseSeen',{'tool':pending})
                completed()
    elif msg.get('id')=='native' and 'result' in msg:
        assert msg['result']=={'decision':'decline'}
        native_main_work()
'''


class TransportTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    async def asyncTearDown(self):
        self.temporary.cleanup()

    def client(self, mode='normal'):
        return AppServer(cwd=self.root, codex_home=self.root / 'auth',
            command=[sys.executable, '-u', '-c', FAKE_SERVER],
            env={**os.environ, 'FAKE_MODE': mode}, request_timeout=2,
            trace_path=self.root / 'trace.jsonl')

    async def test_rpc_dispatcher_independent_of_event_consumer(self):
        async with self.client() as server:
            self.assertIn('--ppblab-protocol', server.command)
            self.assertIn('--strict-config', server.command)
            self.assertEqual(await server.resolve_model(), 'fixture-model')
            self.assertEqual(await server.resolve_model('explicit'), 'explicit')
            results = await asyncio.gather(*(server.request('probe', {'number': n}) for n in range(5)))
            self.assertEqual([r['echo']['number'] for r in results], list(range(5)))
            self.assertEqual((await server.next_event())['method'], 'probe/event')

    async def test_rpc_error_is_informative_and_sanitized(self):
        with self.assertRaises(AppServerError):
            async with self.client('rpc-error'):
                pass

    async def test_prefix_array_replacement_and_clear(self):
        async with self.client() as server:
            for messages in ([{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "first"}]}], []):
                self.assertEqual(await server.set_prefix('thread', messages), {})
                event = await server.next_event()
                self.assertEqual(event['method'], 'fixture/prefixSeen')
                self.assertEqual(event['params']['messages'], messages)

    async def test_prefix_rpc_error_fails(self):
        async with self.client('prefix-error') as server:
            with self.assertRaises(AppServerError):
                await server.set_prefix('thread', [])

    async def test_rpc_error_sanitization(self):
        with self.assertRaises(AppServerError) as caught:
            async with self.client('rpc-error'):
                pass
        message = str(caught.exception)
        self.assertIn('provider unavailable', message)
        self.assertNotIn('private-value', message)
        self.assertNotIn("'secret'", message)
        self.assertNotIn('private-value', (self.root / 'trace.jsonl').read_text())

    async def test_eof_fails_pending_rpc(self):
        async with self.client() as server:
            with self.assertRaises(AppServerError):
                await server.request('fail')

    async def test_close_reaps_process(self):
        async with self.client() as server:
            process = server.process
        self.assertIsNotNone(process.returncode)

    def test_command_override_validated(self):
        from unittest.mock import patch
        with patch.dict(os.environ, {'CODEX_APP_SERVER_COMMAND': '"not argv"'}):
            with self.assertRaises(ValueError):
                server_command()


if __name__ == '__main__':
    unittest.main()
