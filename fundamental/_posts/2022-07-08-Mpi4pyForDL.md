---
layout: post
title: "[CS] 병렬 프로그래밍(3. mpi4py in RL)"
sitemap: false
hide_last_modified: true
---
어떻게 openai spinning up 에서 mpi가 사용되는지

vpg 예를들면

parameter 통일



```
def sync_params(module):
    if num_procs()==1:
        return
    for p in module.parameters():
        p_numpy = p.data.numpy()
        MPI.COMM_WORLD.Bcast(p_numpy, root = root)
```
모델 변수들을 통일한다.

```
def mpi_statistics_scalar(x, with_min_and_max = False):
    """
    MPI processes 들 사이에서 다같이 mean, std를 계산한다.
    """
    x = np.array(x, dtype = np.float32)
    global_sum, global_n = mpi_sum([np.sum(x), len(x)])
```
