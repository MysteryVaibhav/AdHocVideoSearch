import queue as Q

idx = 0;
q = Q.PriorityQueue();
q.put((-0.1,'main'))
q.put((-0.4,'main1'))
q.put((-0.6,'main2'))
q.put((0.7,'main3'))
q.put((4.5,'main4'))
q.put((1.3,'main5'))
q.put((1,'main6'))
while (not q.empty()) and idx < 5:
    idx += 1;
    print(q.get());

import os, subprocess
process_esa = "java -Xmx1024m -cp lib\*;esalib.jar clldsystem.esa.ESAAnalyzer".split(" ")
#Starting esa service
p_esa = subprocess.Popen(process_esa, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, universal_newlines=True)
p_esa.stdin.write('Love\n')
p_esa.stdin.write('Mother\n')
p_esa.stdin.flush()
score = 0
while True:
    try:
        score = float(p_esa.stdout.readline())
        print(score)
        break
    except ValueError:
        pass
p_esa.stdin.write('high\n')
p_esa.stdin.write('fly\n')
p_esa.stdin.flush()
score = 0
while True:
    try:
        score = float(p_esa.stdout.readline())
        print(score)
        break
    except ValueError:
        pass
print("Hey")