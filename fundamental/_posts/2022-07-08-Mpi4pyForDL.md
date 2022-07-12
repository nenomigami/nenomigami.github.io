---
layout: post
title: "[CS] 병렬 프로그래밍(3. mpi4py in RL)"
sitemap: false
hide_last_modified: true
---
이번 포스팅에서는 강화학습 라이브러리에서 mpi 통신을 어떤식으로 활용할 수 있는지 알아보겠습니다. 
대표적인 예로 OpenAI의 오픈소스 프로젝트인 openai baselines의 방식을 vanila policy gradient 코드와 함께 살펴보겠습니다.

# MPI 코드 실행방법

기본적으로 OpenMPI를 사용한 프로그램을 작동하려면 mpiexec 혹은 mpirun 의 명령어로 코드를 작동시켜야합니다. OpenAI에서는 python을 실행하면서 mpi를 작동시키는 방법으로 
subprocess 모듈을 사용했습니다.

```
def mpi_fork(n, bind_to_core=False):
    """
    Args:
        n (int): Number of process to split into.

        bind_to_core (bool): Bind each MPI process to a core.
    """
    if n<=1: #설정 cpu 갯수가 1이하 일때는 종료
        return
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1", # 인텔 mkl thread 수 1 설정
            OMP_NUM_THREADS="1", # OpenMP thread 수 1 설정
            IN_MPI="1" # IN_MPI 라는 환경변수 설정
        )
        args = ["mpirun", "-np", str(n)] # mpi 명령어와 인수 리스트에 할당
        if bind_to_core:
            args += ["-bind-to", "core"]
        args += [sys.executable] + sys.argv # python 응용프로그램 추가
        subprocess.check_call(args, env=env)
        sys.exit()
```
다음과 같은 방법으로 파이썬 내에서 외부 프로세스를 실행하면 직접적으로 디버깅 프로그램이
실행되지 않습니다. 따라서 IDE나 python 기본 디버깅 모듈 외에 다른 방법을 강구해야한다는
단점이 있습니다.

# Vanila Policy Gradient 동작 방식
OpenAI의 Policy Gradient는 흔히 생각하는 방식의 REINFORCE 알고리즘이 아닌 Advantage Actor Critic 알고리즘입니다. Pseudocode는 다음과 같습니다. 


<p align="center">
<img src = "../../assets/mpi_test/AdvPolicyGradientPseudocode.png"
 width="80%" />
</p>
parameter 통일
mpi를 

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

장점 : 

단점 : 디버깅이 어렵다.
