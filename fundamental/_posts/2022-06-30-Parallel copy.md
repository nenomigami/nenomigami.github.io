---
layout: post
title: "[CS] 병렬 프로그래밍(2. mpi4py)"
sitemap: false
hide_last_modified: true
---
어떻게 openai spinning up 에서 mpi가 사용되는지

```
def sync_params(module):
    if num_procs()==1:
        return
    for p in module.parameters():
        p_numpy = p.data.numpy()
        MPI.COMM_WORLD.Bcast(p_numpy, root = root)
```
모델 변수들을 통일한다.