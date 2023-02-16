from multiprocessing import Process, Queue, managers
import time


def worker(id, data, queue, *args):
    tmp1 = time.time()
    if args:
        for i in range(args[0], args[1]):
            data[i] *= 2
        queue.put(0)
    else:
        queue.put([data[x]*2 for x in range(len(data))])


def without_shared_memory():
    print("Without shared memory")
    iterations = 6
    for i in range(2, 3):
        start_time = time.time()
        num_procs = 12
        data = list(range(1, 10**i))
        chunk_size = len(data) // num_procs
        for _ in range(iterations):
            queue = Queue()
            procs = [Process(target=worker,
                             args=(j, data[j*chunk_size:(j+1)*chunk_size],
                                   queue))
                     for j in range(num_procs)]
            for p in procs:
                p.start()

            tmp = 0
            for _ in range(num_procs):
                tmp += sum(queue.get())

            for p in procs:
                p.join()
                p.close()

        end_time = time.time()
        secs_per_iteration = (end_time - start_time) / iterations
        print("data {0:>10,} ints : {1:>6.6f} secs per iteration"
              .format(len(data), secs_per_iteration))
        print(f'sum was: {tmp}')


def with_shared_memory():
    print("With shared memory")
    iterations = 6
    for i in range(2, 3):
        num_procs = 12
        with managers.SharedMemoryManager() as smm:
            start_time = time.time()
            data = smm.ShareableList(range(1, 10**i))
            chunk_size = len(data) // num_procs
            for _ in range(iterations):
                queue = Queue()
                procs = [Process(target=worker,
                                 args=(j, data, queue, j*chunk_size,
                                       (j+1)*chunk_size))
                         for j in range(num_procs)]
                for p in procs:
                    p.start()

                for _ in range(num_procs):
                    queue.get()
                tmp = sum(data)

                for p in procs:
                    p.join()
                    p.close()

            end_time = time.time()
            secs_per_iteration = (end_time - start_time) / iterations
            print("data {0:>10,} ints : {1:>6.6f} secs per iteration"
                  .format(len(data), secs_per_iteration))
            print(f'sum was: {tmp}')


without_shared_memory()
with_shared_memory()