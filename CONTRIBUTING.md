# Contributing

WorldLoop is the hub: protocol, docs, and orchestration. Implementation lands in the component repos (RLinf, vLLM-Omni, …). A change that belongs upstream stays upstream; this repo only records the contract and the current conclusion.

## Pull requests

- One PR, one falsifiable claim (a protocol change, a done-criterion update, or a recorded conclusion).
- `main` takes accepted work only. Feature branches are short-lived.
- PR description states: what is now true, how it was checked, and which repo the code lives in if this PR is docs-only.

## Experiments

Before a GPU run: write the hypothesis and the done criterion. If a negative result would not change the next action, do not run it.

Acceptance is official sim / real-robot eval. World-model-internal success is a collapse sentinel, not a trend.

## AI-assisted contributions

State the tool and how far it participated. "AI-assisted" is not enough.

The author must be able to explain the change without the tool. Issue / PR text may be drafted by a model; a person edits it before posting. No AI-generated images, video, or audio.
