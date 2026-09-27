#!/usr/bin/env python3
"""Send a short peer message to a registered, live local agent.

Runtime registration lives in ignored build/scratch/coordination/sessions.json.
Terminal agents use Orca relay; Codex app-server sessions use their Unix control
socket and turn/steer (active) or turn/start (idle). Credentials stay local.
Does not resume/fork sessions, interrupt work, or modify conversation logs.
After delivery, the recipient must acknowledge in its report: terminal submission
alone is not a receipt. Re-register after restarting a terminal or relay.
"""
import argparse
import json
from pathlib import Path
import socket
import struct
import time

ROOT = Path(__file__).resolve().parents[1]


def proc(pid):
    root = Path('/proc') / str(pid)
    stat = (root / 'stat').read_text().rsplit(')', 1)[1].split()
    return {'comm': (root / 'comm').read_text().strip(),
            'ppid': int(stat[1]), 'start': stat[19],
            'argv': [x.decode() for x in (root / 'cmdline').read_bytes().split(b'\0') if x]}


class Relay:
    def __init__(self, registration):
        p = proc(registration['pid'])
        if p['start'] != registration['start'] or 'relay.js' not in p['argv']:
            raise ValueError('relay registration is stale')
        argv = p['argv']
        sock = Path(argv[argv.index('--sock-path') + 1])
        credential = Path(argv[argv.index('--credential-file') + 1]).read_text().strip()
        version = (sock.parent / '.version').read_text().strip()
        self.socket = socket.socket(socket.AF_UNIX)
        self.socket.settimeout(10)
        self.socket.connect(str(sock))
        self.seq = 0
        self.ack = 0
        self.send({'type': 'orca-relay-handshake', 'version': version,
                   'endpointCredential': credential}, kind=2)
        if self.receive().get('type') != 'orca-relay-handshake-ok':
            raise ValueError('relay handshake failed')

    def send(self, obj, kind=1):
        self.seq += 1
        body = json.dumps(obj).encode()
        self.socket.sendall(struct.pack('>BIII', kind, self.seq, self.ack, len(body)) + body)

    def read(self, size):
        out = b''
        while len(out) < size:
            block = self.socket.recv(size - len(out))
            if not block:
                raise EOFError('relay closed')
            out += block
        return out

    def receive(self):
        kind, seq, _, size = struct.unpack('>BIII', self.read(13))
        if size > 8 * 1024 * 1024:
            raise ValueError('oversized relay frame')
        self.ack = max(self.ack, seq)
        return json.loads(self.read(size)) if size else {}

    def request(self, key, method, params):
        self.send({'jsonrpc': '2.0', 'id': key, 'method': method, 'params': params})
        while True:
            response = self.receive()
            if response.get('id') == key:
                if 'error' in response:
                    raise ValueError(response['error']['message'])
                return response['result']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--to', required=True, choices=['lead', 'claude', 'data', 'codex', 'fable'])
    parser.add_argument('--from-agent', required=True, choices=['lead', 'claude', 'data', 'codex', 'fable'])
    content = parser.add_mutually_exclusive_group(required=True)
    content.add_argument('--text')
    content.add_argument('--file', type=Path)
    parser.add_argument('--check', action='store_true', help='validate destination without sending')
    args = parser.parse_args()
    text = args.text if args.text is not None else args.file.read_text()
    if not text.strip() or len(text) > 1800 or any(ord(c) < 32 and c not in '\n\t' for c in text):
        raise ValueError('message must contain 1–1800 characters and no terminal controls')
    reg = json.loads((ROOT / 'build/scratch/coordination/sessions.json').read_text())
    target = reg['agents'][args.to]
    live = proc(target['pid'])
    if live['start'] != target['start'] or live['comm'] != target['comm']:
        raise ValueError('target agent registration is stale; no input sent')
    message = f'[Peer message from {args.from_agent}; user-authorized coordination] {text}'
    transport = target.get('transport', 'terminal')
    if transport == 'codex-app-server':
        from agent_codex_client import CodexClient
        client = CodexClient(target['socket'], target['pid'])
        try:
            if args.check:
                client.thread(target['session'])
                print(f'VALID destination={args.to} session={target["session"]} transport={transport}')
                return
            turn_id = client.deliver(target['session'], message)
            print(f'SUBMITTED destination={args.to} session={target["session"]} turn={turn_id}; awaiting report acknowledgment')
        finally:
            client.close()
        return
    if transport != 'terminal':
        raise ValueError(f'unsupported agent transport: {transport}')
    relay = Relay(reg['relay'])
    try:
        terminals = relay.request(1, 'pty.listProcesses', {})
        matching = [p for p in terminals if p.get('terminalHandle') == target['terminal']]
        if len(matching) != 1 or matching[0]['title'] != target['comm']:
            raise ValueError('target terminal is not running the registered agent')
        terminal = matching[0]['id']
        states = json.loads(relay.request(2, 'pty.serialize', {'ids': [terminal]}))
        ancestors = set()
        pid = target['pid']
        while pid > 1:
            ancestors.add(pid)
            pid = proc(pid)['ppid']
        if len(states) != 1 or states[0]['pid'] not in ancestors:
            raise ValueError('terminal process is unrelated to target agent')
        if args.check:
            print(f'VALID destination={args.to} session={target["session"]}')
            return
        relay.send({'jsonrpc': '2.0', 'method': 'pty.data',
                    'params': {'id': terminal, 'data': '\x1b[200~' + message + '\x1b[201~'}})
        time.sleep(0.3)
        relay.send({'jsonrpc': '2.0', 'method': 'pty.data', 'params': {'id': terminal, 'data': '\r'}})
        relay.request(3, 'pty.listProcesses', {})
        print(f'SUBMITTED destination={args.to} session={target["session"]}; awaiting report acknowledgment')
    finally:
        relay.socket.close()


if __name__ == '__main__':
    main()
