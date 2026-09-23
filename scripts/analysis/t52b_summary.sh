cd /sjtu/linhang/arena/runs/T52b
until grep -q "^DONE rc=" /sjtu/linhang/arena/runs/jobs/t52b_v2.log; do sleep 20; done
P=/sjtu/linhang/arena/env/m0/bin/python; O=SUMMARY_v2.txt
{ echo "T52b v2 verdicts $(date)  patch md5 $(md5sum kit/170-glm-bcg-prefill.patch | cut -c1-32)"
  for x in "eager170sc_v2tp2 bcg170sc_v2tp2 TP2-scatter:eager-vs-BCG(v2)" "eager170_v2tp2 bcg170_v2tp2 TP2-noscatter:eager-vs-BCG" "eager170_v2tp1 bcg170_v2tp1 TP1:eager-vs-BCG" "eager170_v2tp2 eager170sc_v2tp2 TP2:eager-noscatter-vs-eager-scatter(noise-ref)" "eager170sc_s bcg170sc_s TP2-scatter:eager-vs-BCG(v1-repro)"; do
    set -- $x
    if [ -f $1/correctness.json ] && [ -f $2/correctness.json ]; then echo "== $3"; $P kit/bcg_compare.py $1/correctness.json $2/correctness.json cmp_$1__$2.json | tail -1
    else echo "== $3 MISSING $1 or $2"; fi
  done
  for a in bcg170sc_v2tp2 bcg170_v2tp2 bcg170_v2tp1; do echo "$a graph=True prefill: $(grep 'Prefill batch' $a/server.log | grep -c 'graph: True')/$(grep -c 'Prefill batch' $a/server.log)"; done
} > $O 2>&1
cat $O
