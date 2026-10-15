# Contributing

WorldLoop holds the protocol, the docs, and one reproducible recipe per accepted result. Code that belongs
upstream (RLinf, vLLM-Omni, …) still goes upstream; a recipe pins the upstream commit it ran on and carries
the rest as patches and scripts.

## Recipes

A recipe under `recipes/<name>/` rebuilds a result on a new machine without anything from the machine that
produced it:

- Upstream code is a submodule in `third_party/` at the exact commit; local changes are `patches/*.patch`
  against that commit, checked by tree id after applying.
- Scripts are kept as they ran. Machine paths are rewritten at setup, not edited by hand.
- Every model, checkpoint and dataset the run loads has a public source and a sha256 in `assets.sha256`.
  Small inputs (seed lists, init frames, indices) are committed.
- The README states what was verified, what was not, and what can no longer be regenerated.

## Pull requests

- One PR, one falsifiable claim (a protocol change, a done-criterion update, a recorded conclusion, or a recipe).
- `main` takes accepted work only. Feature branches are short-lived.
- PR description states: what is now true, how it was checked, and which repo the code lives in.

## Experiments

Before a GPU run: write the hypothesis and the done criterion. If a negative result would not change the next action, do not run it.

Acceptance is official sim / real-robot eval. World-model-internal success is a collapse sentinel, not a trend.

## AI-assisted contributions

State the tool and how far it participated. "AI-assisted" is not enough.

The author must be able to explain the change without the tool. Issue / PR text may be drafted by a model; a person edits it before posting. No AI-generated images, video, or audio.
