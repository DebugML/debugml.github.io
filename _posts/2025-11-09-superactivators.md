---
title: "SuperActivators: Reliable Concept Signals from Just a Handful of Tokens"
layout: single
excerpt: "Concept activations are messy—until you focus on the extreme tail. We show that only a tiny set of the highest-activation tokens carries a clean, reliable concept signal."
header:
  overlay_filter: "0.7"
  overlay_image: /assets/images/superactivators/teaser.png
  teaser: /assets/images/superactivators/teaser.png
  actions:
    - label: "Paper (PDF)"
      url: /assets/papers/superactivators.pdf
---

> **TL;DR**: Concept vectors often look noisy and overlap with unrelated tokens. But if you focus on the **top-activation tail**, the signal becomes clear. These sparse **SuperActivator tokens** consistently detect and localize concepts more reliably than full-vector or prompt-based approaches.

## The core idea

Traditional concept analysis aggregates activations from *all* tokens equally—an approach that smooths away the real signal.  
Our key insight: the **extreme high tail** of token activations already contains a clean, separable concept signature.  
Keep only those few tokens, and ignore the rest.

## Why this matters

- **Sharper concept detection** — SuperActivators yield higher concept-presence F1s (up to +14 points over baselines) while using <10 % of tokens  
- **Cleaner attribution maps** — Localizing concepts via these sparse tokens aligns more closely with true spans or regions  
- **Simple and general** — Works across modalities (text + vision), layers, and concept extraction methods with no extra training

## Quick recipe

1. Compute activation scores \(s_c(z)\) for each token \(z\) under concept \(c\)  
2. Estimate a high-tail threshold (e.g., 98th percentile of \(D_c^{\text{out}}\))  
3. Keep tokens \(z\) where \(s_c(z)\) ≥ that threshold — these are your **SuperActivators**

Even though they’re sparse, these tokens capture most of the model’s concept sensitivity while avoiding noisy co-activations.

---

**Takeaway:** Don’t average away the signal — **trust the tail.**


