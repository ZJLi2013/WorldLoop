# TimeLine

```mermaid
gantt
    title WM-as-env to RL to LIBERO bench, closed loop
    dateFormat YYYY-MM-DD
    axisFormat %m/%d
    todayMarker stroke-width:2px,stroke:#d33,opacity:0.6

    section Stage-1 RLinf Recipe
    WorldModelEnv / WorldModelBackend abstraction :done,   s1a, 2026-08-24, 2026-09-01
    Wire in reward model (TopReward)              :done,   s1c, 2026-09-1, 2026-09-10
    Stage-1 done                                  :milestone, m1, 2026-09-12, 0d

    section Stage-2 vLLM-Omni Serving
    RLinf + vLLM-Omni commu protocal              :        s2c, 2026-09-12, 2026-09-17
    Stateless WM support                          :        s2b, 2026-09-17, 2026-09-30
    Stateful WM support                           :        s2a, 2026-09-30, 2026-10-25
    perf optimization and alingment               :        s2d, 2026-10-15, 2026-11-12
    Stage-2 done                                  :milestone, m2, 2026-11-12, 0d
    section Stage-3 evolving
    Agent harness                                 :        s3a, 2026-11-15, 2026-12-24
```



## Stage-1（Aug ~ Oct）— WM-as-env RLInf recipe

* [pr-1518](https://github.com/RLinf/RLinf/pull/1518)
* [pr-1554](https://github.com/RLinf/RLinf/pull/1554)

## Stage-2（Oct ~ Nov）— vLLM-Omni Serving



## Stage-3(Nov ~ Dec) - Self-Evolving Loop 


