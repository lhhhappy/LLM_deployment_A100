# MODEL OUTPUT — T25 first failures

F/O/T/C/P = fast/overall/turn/chain/TPOT; each cell is first failing N, all binding gates.

### MODEL OUTPUT — dev, strict

| Candidate | FCFS | SPF | SPF+D1 | EDF | EDF+D1 | Least-slack | EDF w2 | EDF w2+D1 |
|---|---|---|---|---|---|---|---|---|
| C6 | 6 F+O+T+C | 6 F+C | 6 F+C | 6 F+O+C | 6 F+O+C | 6 F+O+C | 6 F+O+C | 6 F+O+C |
| C17 | 6 F+O+T+C | 6 C | 6 C | 6 F+O+C | 6 F+C | 6 F+O+C | 6 F+C | 6 F+C |
| C18 | 6 F+O+T+C | 6 F+C | 6 F+O+C | 6 F+O+C | 6 F+O+C | 6 F+O+C | 6 F+C | 6 F+O+C |
| C19 | 6 F+O+C | 6 F+C | 6 F+O+C | 6 F+O+C | 6 F+O+C | 6 F+O+C | 6 F+O+C | 6 F+O+C |
| C31 | 6 F+O+T+C | 6 F+C | 6 F+O+C | 6 F+O+C | 6 F+O+C | 6 F+O+C | 6 F+O+C | 6 F+O+C |
| C42 | 6 F+O+T | 6 F+C | 6 F+C | 6 F+O+C | 6 F+O+C | 6 F+O+C | 6 F+O+C | 6 F+C |
| C43 | 6 F+O+T | 6 C | 6 F+C | 6 F+O+C | 6 F+O+C | 6 F+O+C | 6 F+C | 6 F+C |

### MODEL OUTPUT — dev, estimated

| Candidate | FCFS | SPF | SPF+D1 | EDF | EDF+D1 | Least-slack | EDF w2 | EDF w2+D1 |
|---|---|---|---|---|---|---|---|---|
| C6 | 6 F+O | 10 P+F+O | 10 P+F+O+C | 6 F | 6 F | 6 F+O | 6 F | 6 F+C |
| C17 | 6 F+O | 10 P+F+C | 10 P+F+C | 6 F | 10 F+O+C | 6 F+O | 6 C | 10 F+O+C |
| C18 | 6 F+O | 10 P+F+O+C | 10 P+F+O+C | 10 F+O+C | 6 F | 6 F+O | 6 C | 6 C |
| C19 | 6 F+O | 10 F+O+C | 10 F+O+C | 6 F+O | 10 F+O+C | 6 F+O | 6 F+C | 10 F+O+C |
| C31 | 6 F+O | 10 F+O+C | 10 F+C | 6 F | 6 C | 6 O | 6 C | 6 C |
| C42 | 6 F+O | 10 F | 10 F+C | 10 F+O+C | 6 F | 6 O | 10 F+O+C | 10 F+C |
| C43 | 6 F+O | 10 F+C | 10 F+C | 10 F+O+C | 10 F+O+C | 6 F+O | 6 C | 6 C |

### MODEL OUTPUT — formal-mix, strict

| Candidate | FCFS | SPF | SPF+D1 | EDF | EDF+D1 | Least-slack | EDF w2 | EDF w2+D1 |
|---|---|---|---|---|---|---|---|---|
| C6 | 10 F+O+T | 18 F+O | 14 F | 10 F | 10 F | 10 F+O | 10 F | 10 F |
| C17 | 10 F+O | 18 F | 18 F | 14 F+O | 14 F+O | 10 F+O | 10 F | 14 F |
| C18 | 10 F+O | 18 F | 18 F | 10 F | 14 F+O | 10 F+O | 10 F | 14 F+O |
| C19 | 10 F+O | 18 F+O | 14 F | 10 F | 14 F+O | 10 F+O | 10 F+O | 14 F+O+T |
| C31 | 10 F+O | 18 F | 18 F | 10 F+O | 14 F+O | 10 F+O | 10 F | 14 F+O |
| C42 | 10 F+O | 22 F+C | 22 F | 14 F+O | 14 F | 10 F | 14 F+O | 14 F |
| C43 | 10 F+O | 22 F+C | 18 F | 10 F | 14 F+O | 10 F+O | 10 F | 14 F+O |

### MODEL OUTPUT — formal-mix, estimated

| Candidate | FCFS | SPF | SPF+D1 | EDF | EDF+D1 | Least-slack | EDF w2 | EDF w2+D1 |
|---|---|---|---|---|---|---|---|---|
| C6 | 10 F+O | 18 F | 18 F | 10 F | 14 F+O | 10 F+O | 10 F | 14 F+O |
| C17 | 10 F+O | 22 F | 18 F | 14 F+O | 14 F | 10 F | 14 F+O | 14 F |
| C18 | 10 F+O | 18 F | 18 F | 10 F | 14 F+O | 10 F+O | 10 F | 14 F |
| C19 | 10 F+O | 18 F | 18 F | 10 F | 14 F+O | 10 F+O | 10 F | 14 F+O |
| C31 | 10 F+O | 18 F | 18 F | 10 F | 14 F+O | 10 F+O | 10 F | 14 F |
| C42 | 10 F+O | 26 F+C | 22 F | 14 F+O | 14 F | 10 F | 14 F+O | 14 F |
| C43 | 10 F+O | 22 F | 22 F | 10 F | 14 F | 10 F+O | 10 F | 14 F |
