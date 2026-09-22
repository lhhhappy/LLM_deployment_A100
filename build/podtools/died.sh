tail -5 /tmp/ax/runs/011-dev_b111_n6/dev/warmup.log | cut -c1-250
echo ---; tail -3 /tmp/ax/runs/011-dev_b111_n6/dev/preflight.log | cut -c1-250
echo ---SERVER008; grep -n "Traceback\|Error\|Killed\|exception\|SIGTERM\|OOM\|out of memory" /tmp/ax/runs/008-b111_start_probe/server.log | tail -12 | cut -c1-250
echo; tail -25 /tmp/ax/runs/008-b111_start_probe/server.log | cut -c1-250
echo ---NOW; pgrep -af launch_server | head -2; ls -la /tmp/ax/runs/012-dev_b111_n10/; tail -5 /tmp/ax/runs/012-dev_b111_n10/job.log
dmesg 2>/dev/null | tail -5; free -g | head -2
