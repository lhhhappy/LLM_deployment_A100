cd /tmp/ax/runs/008-b111_start_probe
grep 'TP0\]' server.log | grep -v 'capped' | awk '/17:59:5/{f=1} f' | cut -c1-230 | head -45
echo ---; grep -i 'compil\|autotun\|jit' server.log | cut -c1-200 | sort | uniq -c | sort -rn | head -15
