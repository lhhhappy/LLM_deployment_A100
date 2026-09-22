out=/tmp/ax/prof_$(date +%H%M%S); mkdir -p $out
curl -s -m 60 -X POST localhost:30000/start_profile -H 'Content-Type: application/json' -d "{\"output_dir\":\"$out\",\"num_steps\":12,\"activities\":[\"GPU\",\"CPU\"]}"; echo
for i in $(seq 1 40); do n=$(ls $out 2>/dev/null | wc -l); [ $n -ge 8 ] && break; sleep 3; done
sleep 5; ls -la $out | head; echo $out > /tmp/ax/prof_last
