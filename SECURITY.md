# SECURITY.md

## Overview
This repository contains the reference implementation of TEA, a targeted hard-label adversarial attack against image classifiers. The work’s primary intent is scientific—studying robustness, evaluating defences, and understanding failure modes of modern vision models. However, the same techniques could be misused. This document sets out dual-use risks, safe-use expectations, and reporting channels.

## Dual-Use Risk Assessment
Potential misuse scenarios include:
- Degrading the performance of cloud-hosted vision APIs (availability/integrity attacks)
- Spoofing or weakening biometric recognition systems (e.g., face, fingerprint) without consent
- Bypassing safety filters in downstream applications that rely on image classifiers
- Automating targeted misclassification to facilitate fraud or evasion

## Intended Use
- Academic and industrial research on robustness, attack/defence co-design, and risk assessment
- Reproducible evaluation of targeted hard-label attacks under controlled settings

Out of scope:
- Use on real-world biometric or security-critical systems without explicit authorization
- Any activity that violates terms of service, local laws, or privacy/ethics approvals

## Release Rationale
The core algorithm is easily reproducible from the paper’s description; gating or delaying the code would provide little additional protection while hindering reproducibility. 

## Model-Weight Policy
- TEA uses only publicly available checkpoints (e.g., ResNet, VGG, ViT).
- No new or proprietary model weights are released in this repository.

## Reporting Security Concerns or Vulnerabilities
If you identify:
- A vulnerability in this code that could cause unintended exposure or harm
- A plausible pathway to high-risk misuse not described here
- Results that may impact a third-party system’s security

Report first:
1) Email the contact address listed in this repository (in README).

Please include:
- A clear description, minimal reproduction, and potential impact
- Affected versions/commits and environment details
- Suggested remediation or mitigating steps if available

We follow coordinated disclosure principles and will acknowledge receipt as promptly as possible.

## License and Usage Terms
- Default code license: MIT (see LICENSE).

## Changes to This Policy
This SECURITY.md may evolve as risks and community practices change.
