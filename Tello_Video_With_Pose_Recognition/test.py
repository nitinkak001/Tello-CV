import queue
q1 = queue.Queue(4)
q1.put(1.5)
q1.put(2.5)
q1.put(3.0)
q1.put(4.0)

print(q1.get())