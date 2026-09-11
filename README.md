# WorldLoop

WorldLoop is an embodied-model self-evolution framework: SOTA world models act as the env simulator for online RL of embodied policies (VLA and similar).

This repo is the hub — protocol, docs, and orchestration. Implementation lands in the component repos.

The value of a WM is not that it is cheaper or faster than a conventional simulator. Sim coverage is capped by what humans can model (assets, contact, lighting, scenes). A WM is learned from video; its boundary follows the data. The deployment setting is the real world, where there is no sim benchmark.

A WM has no correctness guarantee. The path is: first measure how much utility is lost when a WM stands in for sim on a benchmark that still has a simulator, then push to settings with no sim. **LIBERO is that calibration field, not a proof of value.**

![architecture](./images/worldloop-architecture.png)

## Technical constraints

WM-as-env and sim-as-env are different constraint sets: no physical state, no oracle, interaction cost is compute. Four locked choices:

1. **On-policy, critic-free (GRPO).** A WM has no physical state; a critic sits on a softer foundation than under sim. The same visual observation can map to different true configurations.
2. **Reward is external and not trained** (zero-shot VLM for progress). A WM has no oracle; the scorer is the only piece that transfers across benchmarks and WMs.
3. **Sample volume comes from cheaper generation** (diffusion steps, KIR, serving batching and weight sharing), not from replaying old samples. Replay multiplies WM dynamics bias by reuse count.
4. **Acceptance is always official eval on real sim / real robot.** WM-internal success does not measure trend; it is only a collapse sentinel.

## Components

| Project | Role in the loop |
|---|---|
| vLLM-Omni | Rollout / serving / world-model substrate |
| verl-omni | Diffusion / generative RL engine |
| RLinf | Embodied RL reference and baseline |
| WMForge | WM → benchmark adaptation; produces env-ready bundles |

## Current bar

**Stage-2 done:** same WM, same policy; in-process vs remote official eval stay within the noise floor. This bar is correctness only. Serving performance and autoregressive WM sessions are what it unlocks, not the criterion.

Timeline: [docs/roadmap.md](docs/roadmap.md). How to contribute: [CONTRIBUTING.md](CONTRIBUTING.md).
