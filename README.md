# Emergent Money Agents

This repository contains the original implementation of an agent-based economic simulation and the accompanying PDF documentation.

## Contents

- `EmergentMoney.pdf` &ndash; Document describing the concepts and rules behind the simulation.
- `adamsmithfeb17B.c` &ndash; Legacy C source code (written around 2010) implementing the simulation from the PDF. It depends on `stdheaders.h`, which is not included here.
- `LICENSE` &ndash; MIT license covering the repository contents.

## Purpose of this Repository

The legacy C code does not directly reference the PDF, but it is an early attempt at turning the ideas in *Emergent Money* into a working program. Today it serves primarily as a historical reference. We do **not** aim to compile or run this version as-is; instead, both the PDF and the C code are used as guidance while rebuilding the simulation with modern tools.

## Modernisation Goals

The upcoming implementation will:

- Be rewritten with a modular structure and a visual user interface.
- Leverage parallel computing on RTX 3090/4090 GPUs.
- Provide hooks for experiments in policy, economic games and educational scenarios.

As this work progresses, the repository will house the new source code alongside the original files for reference.

