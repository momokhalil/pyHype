import mpi4py as mpi

num_BLK = 23
num_procs = mpi.MPI.COMM_WORLD.Get_size()
per_cpu = num_BLK // num_procs
rem = num_BLK % num_procs

dist = {}
counter = 0
for n in range(num_procs):
    dist[n] = [counter + i for i in range(per_cpu + 1 if n < rem else per_cpu)]
    counter += len(dist[n])

print(dist)
