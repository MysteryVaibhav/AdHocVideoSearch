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