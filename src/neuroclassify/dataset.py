from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class CorpusSample:
    text: str
    label: str

class EnterpriseCorpus:
    """Enterprise text & telemetry dataset across 4 operational domains."""

    RAW_SAMPLES: List[Tuple[str, str]] = [
        # --- 1. SECURITY_INCIDENT ---
        ("ALERT: Repeated SSH authentication failures from external IP 192.168.1.100 exceeding brute force threshold.", "SECURITY_INCIDENT"),
        ("Privilege escalation anomaly: service account 'ci-runner' attempted unauthorized sudo role assumption.", "SECURITY_INCIDENT"),
        ("Inbound DDoS detected: incoming SYN packet volume surpassed 60 Gbps at perimeter edge router.", "SECURITY_INCIDENT"),
        ("Vulnerability scanner reported CVE-2024-3094 buffer overflow in liblzma dependency.", "SECURITY_INCIDENT"),
        ("Suspicious egress transfer: 42 GB of encrypted database dump transmitted to unverified external endpoint.", "SECURITY_INCIDENT"),
        ("WAF blocked SQL injection attempt in HTTP authorization header with payload UNION SELECT null.", "SECURITY_INCIDENT"),
        ("Cryptographic key compromised: private TLS certificate accessed by non-root daemon process.", "SECURITY_INCIDENT"),
        ("Malware signature detected in temporary upload directory: Trojan.Linux.Generic hash mismatch.", "SECURITY_INCIDENT"),

        # --- 2. INFRASTRUCTURE_ALERT ---
        ("CRITICAL: Redis master node cluster memory consumption reached 96.8% with eviction policy failing.", "INFRASTRUCTURE_ALERT"),
        ("Kubernetes node worker-pool-3 entered NotReady status following kernel OOM killer termination.", "INFRASTRUCTURE_ALERT"),
        ("Database query connection pool exhausted: 250 of 250 slots active with 42 clients waiting.", "INFRASTRUCTURE_ALERT"),
        ("High disk I/O latency on NVMe volume /dev/nvme0n1: average wait time spiked above 450ms.", "INFRASTRUCTURE_ALERT"),
        ("HTTP 504 Gateway Timeout rate exceeded 8.2% across European ingress gateway endpoints.", "INFRASTRUCTURE_ALERT"),
        ("PostgreSQL replica replication lag behind primary database exceeded 120 seconds.", "INFRASTRUCTURE_ALERT"),
        ("Elasticsearch shard allocation failed: insufficient unassigned storage remaining on cluster datanodes.", "INFRASTRUCTURE_ALERT"),
        ("DNS resolution failures detected across internal CoreDNS service discovery pods.", "INFRASTRUCTURE_ALERT"),

        # --- 3. LEGAL_COMPLIANCE ---
        ("This Master Services Agreement is entered into between Acme Corp and the designated enterprise customer.", "LEGAL_COMPLIANCE"),
        ("Under Section 14, total indemnification liability shall not exceed the fees paid in the prior twelve months.", "LEGAL_COMPLIANCE"),
        ("Both parties agree that confidential trade secrets, algorithms, and technical specs remain proprietary.", "LEGAL_COMPLIANCE"),
        ("All customer personal data processing must strictly adhere to European GDPR and California CCPA guidelines.", "LEGAL_COMPLIANCE"),
        ("Arbitration of commercial disputes shall be administered under international ICC rules in London.", "LEGAL_COMPLIANCE"),
        ("The contractor assigns all patentable inventions and software copyright conceived during the project.", "LEGAL_COMPLIANCE"),
        ("Termination for convenience requires ninety days written formal notice delivered to corporate counsel.", "LEGAL_COMPLIANCE"),
        ("The non-solicitation covenant prohibits recruiting technical employees for eighteen months post-termination.", "LEGAL_COMPLIANCE"),

        # --- 4. FINANCIAL_BILLING ---
        ("Invoice #INV-2024-8841 due within thirty days: total payable $18,450.00 including international VAT.", "FINANCIAL_BILLING"),
        ("Payment settlement notification: wire transfer of $45,000.00 confirmed to designated corporate escrow bank.", "FINANCIAL_BILLING"),
        ("Quarterly accounts receivable reconciliation indicates three enterprise accounts are past agreed net-60 terms.", "FINANCIAL_BILLING"),
        ("Stripe subscription billing failure: credit card ending in 8841 declined with insufficient funds code.", "FINANCIAL_BILLING"),
        ("Credit memo #CM-102 issued for cloud SLA outage rebate: $1,250.00 credited to customer ledger.", "FINANCIAL_BILLING"),
        ("Remittance advice: monthly SaaS license fee for 250 seats processed through ACH corporate automated clearing.", "FINANCIAL_BILLING"),
        ("Sales tax audit discrepancy: uncollected jurisdiction tax for digital service downloads in fiscal Q2.", "FINANCIAL_BILLING"),
        ("Dispute notice: customer flagged duplicate billing transaction for compute cluster instances.", "FINANCIAL_BILLING")
    ]

    @classmethod
    def load(cls) -> Tuple[List[str], List[str]]:
        texts = [s[0] for s in cls.RAW_SAMPLES]
        labels = [s[1] for s in cls.RAW_SAMPLES]
        return texts, labels
