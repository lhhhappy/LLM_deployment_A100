"""Local Codex app-server peer transport (Unix WebSocket, standard library).

Connects to an existing server and thread. Never resumes, forks, interrupts, or
edits transcript files. The socket's peer PID must match the registration.
Protocol: https://learn.chatgpt.com/docs/app-server
"""

import base64
import hashlib
import json
import os
from pathlib import Path
import socket
import struct


def control_socket(pid):
    """Locate the listening Unix socket actually held by this app-server PID."""
    root = Path('/proc') / str(pid)
    argv = (root / 'cmdline').read_bytes().split(b'\0')
    if b'app-server' not in argv:
        raise ValueError('registered process is not a Codex app-server')
    inodes = set()
    for fd in (root / 'fd').iterdir():
        try:
            target = str(fd.readlink())
        except OSError:
            continue
        if target.startswith('socket:['):
            inodes.add(target[8:-1])
    paths = set()
    for line in Path('/proc/net/unix').read_text().splitlines()[1:]:
        fields = line.split()
        if (len(fields) == 8 and fields[6] in inodes
                and int(fields[3], 16) & 0x10000 and fields[7].startswith('/')):
            paths.add(fields[7])
    if len(paths) != 1:
        raise ValueError(f'expected one app-server control socket, found {len(paths)}')
    return paths.pop()


class CodexClient:
    MAX_MESSAGE = 8 * 1024 * 1024

    def __init__(self, path, pid):
        self.socket = socket.socket(socket.AF_UNIX)
        self.socket.settimeout(10)
        self.buffer = bytearray()
        self.seq = 0
        try:
            self.socket.connect(path)
            peer_pid, peer_uid, _ = struct.unpack(
                '3i', self.socket.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
            if peer_pid != pid or peer_uid != os.getuid():
                raise ValueError('control socket does not belong to registered app-server')
            key = base64.b64encode(os.urandom(16)).decode()
            self.socket.sendall((
                'GET / HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\n'
                'Connection: Upgrade\r\nSec-WebSocket-Version: 13\r\n'
                f'Sec-WebSocket-Key: {key}\r\n\r\n').encode())
            while b'\r\n\r\n' not in self.buffer:
                block = self.socket.recv(4096)
                if not block:
                    raise EOFError('app-server closed during WebSocket handshake')
                self.buffer.extend(block)
                if len(self.buffer) > 16384:
                    raise ValueError('oversized WebSocket handshake')
            header, rest = self.buffer.split(b'\r\n\r\n', 1)
            self.buffer = bytearray(rest)
            lines = header.decode('ascii').split('\r\n')
            headers = dict(line.split(':', 1) for line in lines[1:])
            headers = {k.lower(): v.strip() for k, v in headers.items()}
            expected = base64.b64encode(hashlib.sha1(
                (key + '258EAFA5-E914-47DA-95CA-C5AB0DC85B11').encode()).digest()).decode()
            if (lines[0].split()[1] != '101'
                    or headers.get('sec-websocket-accept') != expected):
                raise ValueError('app-server WebSocket upgrade rejected')
            self.request('initialize', {
                'clientInfo': {'name': 'arena_peer', 'version': '1.0'},
                'capabilities': {'experimentalApi': True}})
            self.send({'method': 'initialized', 'params': {}})
        except Exception:
            self.socket.close()
            raise

    def read(self, size):
        while len(self.buffer) < size:
            block = self.socket.recv(min(65536, size - len(self.buffer)))
            if not block:
                raise EOFError('app-server closed')
            self.buffer.extend(block)
        out = bytes(self.buffer[:size])
        del self.buffer[:size]
        return out

    def frame(self, data, opcode=1):
        mask = os.urandom(4)
        size = len(data)
        header = bytes([0x80 | opcode])
        if size < 126:
            header += bytes([0x80 | size])
        elif size < 65536:
            header += bytes([0x80 | 126]) + struct.pack('>H', size)
        else:
            header += bytes([0x80 | 127]) + struct.pack('>Q', size)
        masked = bytes(value ^ mask[i % 4] for i, value in enumerate(data))
        self.socket.sendall(header + mask + masked)

    def send(self, obj):
        self.frame(json.dumps(obj).encode())

    def receive(self):
        parts = bytearray()
        while True:
            first, second = self.read(2)
            opcode, size = first & 0xF, second & 0x7F
            if first & 0x70 or second & 0x80:
                raise ValueError('unsupported server WebSocket frame')
            if size == 126:
                size = struct.unpack('>H', self.read(2))[0]
            elif size == 127:
                size = struct.unpack('>Q', self.read(8))[0]
            if size + len(parts) > self.MAX_MESSAGE:
                raise ValueError('oversized app-server message')
            data = self.read(size)
            if opcode == 8:
                raise EOFError('app-server WebSocket closed')
            if opcode == 9:
                self.frame(data, opcode=10)
                continue
            if opcode == 10:
                continue
            if opcode not in (0, 1):
                raise ValueError('expected app-server JSON text frame')
            parts.extend(data)
            if first & 0x80:
                return json.loads(parts)

    def request(self, method, params):
        self.seq += 1
        key = self.seq
        self.send({'id': key, 'method': method, 'params': params})
        while True:
            message = self.receive()
            if message.get('id') == key:
                if 'error' in message:
                    raise ValueError(message['error']['message'])
                return message['result']

    def thread(self, thread_id):
        thread = self.request('thread/read', {
            'threadId': thread_id, 'includeTurns': False})['thread']
        if thread['id'] != thread_id or thread['status']['type'] not in ('active', 'idle'):
            raise ValueError('target thread is not loaded and available on this server')
        return thread

    def deliver(self, thread_id, text):
        thread = self.thread(thread_id)
        params = {'threadId': thread_id, 'input': [{'type': 'text', 'text': text}]}
        if thread['status']['type'] == 'active':
            turns = self.request('thread/turns/list', {
                'threadId': thread_id, 'limit': 1, 'sortDirection': 'desc',
                'itemsView': 'notLoaded'})['data']
            if len(turns) != 1 or turns[0]['status'] != 'inProgress':
                raise ValueError('active turn changed; retry delivery after checking target')
            params['expectedTurnId'] = turns[0]['id']
            result = self.request('turn/steer', params)
            return result['turnId']
        return self.request('turn/start', params)['turn']['id']

    def close(self):
        self.socket.close()
