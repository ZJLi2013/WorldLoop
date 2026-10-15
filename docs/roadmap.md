# Timeline

![timeline](../images/timeline.png)

> Source `images/timeline.mmd`; regenerate with
> `npx @mermaid-js/mermaid-cli -i images/timeline.mmd -o images/timeline.png -b white -s 2`.

## Stage-1 (Aug 24 – Sep 14) — WM-as-env recipe in RLinf · LIBERO — done

* `WorldModelEnv` / `WorldModelBackend`: [#1518](https://github.com/RLinf/RLinf/pull/1518) (merged)
* [#1554](https://github.com/RLinf/RLinf/pull/1554)
* Open: the LIBERO success metric does not rise with training steps.

## Stage-2 (Sep 7 – Oct 13) — WM-as-env calibration on RoboTwin · BWM — done

* Stateless remote WM serving: [#1567](https://github.com/RLinf/RLinf/pull/1567) (draft)
* Failure-data WM fine-tuning, frozen T5 reward, corrected PPO: 5 / 5 seeds above SFT, mean +9.7 pp.
  Pipeline: [`recipes/robotwin-bwm-ppo`](../recipes/robotwin-bwm-ppo/README.md)
* Parked: generation speed (rollout is ~97% of a training step), stateful WM.

## Stage-3 (from Sep 30) — RobotRSI: failure recovery and skill evolution

* Agent harness; first deliverable is a baseline / recovery comparison with a failure taxonomy.
