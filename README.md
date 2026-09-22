# Sustainable AI Lifecycle Auditor

> Status: Work in Progress

This project estimates the complete lifecycle energy and carbon
impact of AI systems and recommends the most sustainable model
that satisfies user-defined accuracy and latency requirements.

## Current progress

- [x] Role 1: Lifecycle carbon calculator
- [ ] Role 2: Real model experiments and measurements
- [x] Role 3: Hybrid local-AI decision engine
- [ ] Role 4: Website and visualization
- [ ] Final end-to-end integration

## Important notice

Some CSV values are currently demonstration assumptions.
They will be replaced with sourced factors and measured
experimental results before final submission.

## Components

### Carbon engine

Calculates emissions from:

- Training
- Inference
- Storage
- Networking
- Retraining
- Hardware manufacturing

### Decision engine

Uses:

- Local Qwen3 through Ollama
- Deterministic numeric validation
- Accuracy and latency constraints
- Minimum-carbon model selection
- Explainable recommendations

## Local AI requirement

Install Ollama and download the model:

```bash
ollama pull qwen3:1.7b
