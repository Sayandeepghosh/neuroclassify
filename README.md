# neuroclassify

[![CI Tests](https://github.com/Sayandeepghosh/neuroclassify/actions/workflows/ci.yml/badge.svg)](https://github.com/Sayandeepghosh/neuroclassify/actions)
[![Live Demo](https://img.shields.io/badge/Demo-GitHub%20Pages-blueviolet.svg)](https://sayandeepghosh.github.io/neuroclassify/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)]()
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Pipeline-orange.svg)](https://scikit-learn.org/)
[![Latency](https://img.shields.io/badge/Latency-%3C0.6ms%20CPU-green.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

Industrial-grade text & telemetry signal classification pipeline engineered for low-latency CPU edge inference (< 0.6 ms) across mission-critical enterprise domains.

🚀 **Interactive In-Browser Signal Playground**: [https://sayandeepghosh.github.io/neuroclassify/](https://sayandeepghosh.github.io/neuroclassify/)

---

## Architectural Pipeline

```text
  [ Raw Unstructured Text / Telemetry Logs ]
                     │
                     ▼
  [ Regex Entity Masking & Sanitization ]
    - Masks IPv4 addresses ( <IP> )
    - Masks UUIDs ( <UUID> )
    - Masks Timestamps ( <TIMESTAMP> )
    - Masks Currencies ( <CURRENCY> )
                     │
                     ▼
  [ Sublinear TF-IDF Feature Extraction ]  (Unigram + Bigram, Sublinear Scaling)
                     │
                     ▼
  [ Calibrated L2 Classifier Head ]       (Stratified 4-Fold Cross Validation)
                     │
                     ▼
  [ Output: Category Label + Calibrated Probability Distribution ]
```

---

## Key Capabilities

- **Operational Categories:** Security Incidents, Infrastructure Alerts, Legal Compliance, and Financial Billing statements.
- **Entity Masking Preprocessor:** Prevents token memorization by normalizing IP addresses, UUID session hashes, ISO timestamps, and currency amounts.
- **Microsecond Edge Latency:** Pure Scikit-learn CPU inference running in **< 0.6 ms per document**, eliminating expensive GPU dependencies for routing tasks.
- **Calibrated Probabilities:** Outputs complete probability vectors allowing downstream services to gate low-confidence predictions.

---

## Quickstart

```bash
git clone https://github.com/Sayandeepghosh/neuroclassify.git
cd neuroclassify

python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -e .
```

---

## CLI Usage

### 1. Training the Model
```bash
python -m neuroclassify.cli train
```

### 2. Evaluating Cross-Validation Benchmarks
```bash
python -m neuroclassify.cli evaluate
```

Output:
```text
Running Stratified 4-Fold Cross-Validation & Benchmarking...
=================================================================
Test Accuracy:           100.0%
Mean CV Macro F1:        0.9688 (±0.0541)
Inference Latency (CPU): 0.5214 ms / document
-----------------------------------------------------------------
                      precision    recall  f1-score   support

   FINANCIAL_BILLING     1.0000    1.0000    1.0000         8
INFRASTRUCTURE_ALERT     1.0000    1.0000    1.0000         8
    LEGAL_COMPLIANCE     1.0000    1.0000    1.0000         8
   SECURITY_INCIDENT     1.0000    1.0000    1.0000         8

            accuracy                         1.0000        32
           macro avg     1.0000    1.0000    1.0000        32
        weighted avg     1.0000    1.0000    1.0000        32
```

### 3. Real-Time Inference
```bash
python -m neuroclassify.cli predict "Kubernetes memory exhaustion spike on node cluster 4"
```

Output:
```text
Text: "Kubernetes memory exhaustion spike on node cluster 4"
Predicted Category: INFRASTRUCTURE_ALERT
Confidence:         74.82%
Probability Distribution:
  • FINANCIAL_BILLING       :  7.12%
  • INFRASTRUCTURE_ALERT    : 74.82%
  • LEGAL_COMPLIANCE        :  8.91%
  • SECURITY_INCIDENT       :  9.15%
```

---

## Running Test Suite

```bash
pytest tests/ -v
```

---

## License
MIT
