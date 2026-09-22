d=/tmp/ax/runs/$1; tail -8 $d/job.log | cut -c1-300; echo ---; ls $d/dev 2>/dev/null; for f in $d/dev/*.log; do echo "== $f"; tail -15 $f | cut -c1-300; done
