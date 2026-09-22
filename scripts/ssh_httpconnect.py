import socket,sys,os,selectors
h,p=sys.argv[1],sys.argv[2]
s=socket.create_connection(('127.0.0.1',7890))
s.sendall(f'CONNECT {h}:{p} HTTP/1.1\r\nHost: {h}:{p}\r\n\r\n'.encode())
buf=b''
while b'\r\n\r\n' not in buf:
    c=s.recv(1)
    if not c: sys.exit(1)
    buf+=c
if b' 200' not in buf.split(b'\r\n')[0]: sys.stderr.write(buf.decode()); sys.exit(1)
s.setblocking(True)
sel=selectors.DefaultSelector()
sel.register(0,selectors.EVENT_READ,'in'); sel.register(s,selectors.EVENT_READ,'sock')
stdin_open=True
while True:
    for key,_ in sel.select():
        if key.data=='in':
            d=os.read(0,262144)
            if not d:
                sel.unregister(0); stdin_open=False
                try: s.shutdown(socket.SHUT_WR)
                except OSError: pass
            else: s.sendall(d)
        else:
            d=s.recv(262144)
            if not d: sys.exit(0)
            v=memoryview(d)
            while v:
                n=os.write(1,v); v=v[n:]
