#!/usr/bin/env python3
"""Register this agent in build/scratch/coordination/sessions.json.

Usage: python3 scripts/agent_register.py --name fable --pid <agent pid> --session <session id>
The terminal handle is found through the relay: the one terminal whose process is an ancestor of the agent pid.
For a Codex session without a PTY, use --transport codex-app-server with its
app-server PID. The live thread is checked through the owned control socket.
Credentials are read from the relay process only during the handshake and are never written or printed.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import tempfile

from agent_message import ROOT, Relay, proc


def save_registration(path, name, value):
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        latest = json.loads(path.read_text())
        latest['agents'][name] = value
        fd, temp = tempfile.mkstemp(dir=path.parent, prefix='.sessions-')
        try:
            with os.fdopen(fd, 'w') as stream:
                stream.write(json.dumps(latest, indent=1) + '\n')
            os.replace(temp, path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--session', required=True)
    parser.add_argument('--transport', choices=['terminal', 'codex-app-server'], default='terminal')
    args = parser.parse_args()
    path = ROOT / 'build/scratch/coordination/sessions.json'
    reg = json.loads(path.read_text())
    me = proc(args.pid)
    identity = {'pid': args.pid, 'start': me['start'], 'comm': me['comm'], 'session': args.session}
    if args.transport == 'codex-app-server':
        from agent_codex_client import CodexClient, control_socket
        sock = control_socket(args.pid)
        client = CodexClient(sock, args.pid)
        try:
            client.thread(args.session)
        finally:
            client.close()
        save_registration(path, args.name, {
            **identity, 'transport': args.transport, 'socket': sock})
        print(f'REGISTERED name={args.name} pid={args.pid} transport={args.transport} session={args.session}')
        return
    ancestors = set()
    pid = args.pid
    while pid > 1:
        ancestors.add(pid)
        pid = proc(pid)['ppid']
    relay = Relay(reg['relay'])
    try:
        terminals = relay.request(1, 'pty.listProcesses', {})
        handles = []
        for t in terminals:
            states = json.loads(relay.request(2, 'pty.serialize', {'ids': [t['id']]}))
            if len(states) == 1 and states[0]['pid'] in ancestors:
                handles.append(t)
    finally:
        relay.socket.close()
    if len(handles) != 1:
        raise SystemExit(f'expected exactly one terminal owning pid {args.pid}, found {len(handles)}')
    t = handles[0]
    save_registration(path, args.name, {**identity, 'terminal': t['terminalHandle']})
    print(f'REGISTERED name={args.name} pid={args.pid} comm={me["comm"]} terminal={t["terminalHandle"]} title={t.get("title")}')


if __name__ == '__main__':
    main()
