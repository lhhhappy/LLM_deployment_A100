#!/usr/bin/env python3
"""Register (or refresh) this agent terminal in build/scratch/coordination/sessions.json.

Usage: python3 scripts/agent_register.py --name fable --pid <agent pid> --session <session id>
The terminal handle is found through the relay: the one terminal whose process is an ancestor of the agent pid.
Credentials are read from the relay process only during the handshake and are never written or printed.
"""
import argparse
import json
from pathlib import Path

from agent_message import ROOT, Relay, proc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--session', required=True)
    args = parser.parse_args()
    path = ROOT / 'build/scratch/coordination/sessions.json'
    reg = json.loads(path.read_text())
    me = proc(args.pid)
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
    reg['agents'][args.name] = {'pid': args.pid, 'start': me['start'], 'comm': me['comm'],
                                'session': args.session, 'terminal': t['terminalHandle']}
    path.write_text(json.dumps(reg, indent=1) + '\n')
    print(f'REGISTERED name={args.name} pid={args.pid} comm={me["comm"]} terminal={t["terminalHandle"]} title={t.get("title")}')


if __name__ == '__main__':
    main()
