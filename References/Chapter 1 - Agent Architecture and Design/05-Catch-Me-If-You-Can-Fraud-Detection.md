# Catch Me If You Can: A Multi-Agent Framework for Financial Fraud Detection

**Source:** https://ieeexplore.ieee.org/abstract/document/11113154/

## Overview

"Catch Me If You Can" is a Multi-Agent Framework designed to generate synthetic datasets and simulate various types of fraudulent behavior in financial systems, including:
- Anti-money laundering (AML)
- Credit card fraud
- Bot attacks
- Malicious traffic patterns

## Framework Architecture

### Core Agent Types

The framework comprises two core agent types working in adversarial interaction:

1. **Detector Agents** - Trained to identify suspicious patterns and anomalies
2. **Transaction Agents** - Participants that include:
   - Legitimate transaction participants
   - Fraud agents employing evasion strategies

### Coevolutionary Dynamics

- **Detectors** iteratively refine their detection strategies as new fraud patterns emerge
- **Fraud agents** evolve adaptive tactics to disguise illicit activities
- Creates an **adversarial coevolutionary environment** simulating real-world fraud cat-and-mouse games

## Problem Statement

Modern fraud is:
- **Non-stationary** - continuously evolving in real time
- **Sophisticated** - driven by bots, social engineering, synthetic identities
- **Adversarial** - designed to evade detection mechanisms

## Key Advantages

1. **Synthetic Data Generation** - Creates diverse, realistic fraud scenarios for training
2. **Adaptive Testing** - Evaluates detection systems against evolving attack strategies
3. **Scalability** - Can simulate complex financial networks and transaction patterns
4. **Realism** - Incorporates authentic fraud tactics and evasion techniques

## Related Research

The framework connects to broader research on:
- Multi-agent systems for money laundering detection
- "Multiverse Simulation" frameworks using multi-agent systems to generate AML training datasets
- Creation of virtual worlds with varying illicit activity parameters

## Applications

- **Training fraud detection systems** with diverse, realistic scenarios
- **Stress testing** detection mechanisms against advanced adversaries
- **Understanding** fraud evolution and adaptation patterns
- **Policy development** for financial institutions
