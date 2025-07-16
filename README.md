# Emergent Money Agents

This repository contains the original implementation of an agent-based economic simulation and the accompanying PDF documentation.

## Contents

- `EmergentMoney.pdf` &ndash; Document describing the concepts and rules behind the simulation.
- `Working_Legacy_Code_Reference.pdf` &ndash; Nwely annotated Legacy C source code (code written around 2010) implementing the simulation from the EmergentMoney.PDF. It depends on `stdheaders.h`, which is not included here.
- `LICENSE` &ndash; MIT license covering the repository contents.

## Purpose of this Repository

The legacy C code in Working_Legacy_Code_Reference.pdf is annotated to be in sync with the PDF, and the code has been used to produce the reported results found in *Emergent Money*. Today it serves primarily as a historical reference. We do **not** aim to compile or run this version as-is; instead, both the PDF-report and the annotated C code are used as guidance while rebuilding the simulation with modern tools.

## Modernisation Goals

The upcoming implementation will:

- Be rewritten with a modular structure and a visual user interface.
- Leverage parallel computing on RTX 3090/4090 GPUs.
- Provide hooks for experiments in policy, economic games and educational scenarios.

As this work progresses, the repository will house the new source code alongside the original files for reference.

