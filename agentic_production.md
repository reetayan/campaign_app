

## Architecture Diagram

```mermaid
flowchart TB

    subgraph Sources[Enterprise Signal Sources]
        S1[Application Logs]
        S2[Pipeline Logs]
        S3[Infra Metrics]
        S4[Model Monitoring Metrics]
        S5[Feature Distributions]
        S6[Git / Deployment Metadata]
        S7[Incident History / Runbooks]
    end

    subgraph Ingestion[Monitoring and Ingestion Layer]
        I1[Log Collectors]
        I2[Metrics Collector]
        I3[Feature Snapshot Loader]
        I4[Incident Knowledge Base]
    end

    subgraph AgenticCore[Agentic AI Core]
        A1[Detection Agent<br/>ML + Statistical Monitoring]
        A2[Context Builder Agent<br/>Signal Aggregation]
        A3[RCA Agent<br/>LLM Reasoning + Structured Evidence]
        A4[Recommendation Agent<br/>LLM + Similar Incident Retrieval]
        A5[Execution Agent<br/>Automation + Human Approval]
    end

    subgraph MLServices[ML and Intelligence Services]
        M1[Anomaly Detection Models]
        M2[Drift Detection Engine<br/>PSI / KL / KS]
        M3[Vector DB for Past Incidents]
        M4[LLM Orchestration Layer]
    end

    subgraph Actions[Resolution and Action Layer]
        R1[Trigger Job Retry]
        R2[Rollback Model / Deployment]
        R3[Create JIRA / Incident Ticket]
        R4[Send Slack / Email Alert]
        R5[Escalate to Engineer]
    end

    subgraph Governance[Governance and Feedback Layer]
        G1[Human-in-the-Loop Approval]
        G2[Audit Logs]
        G3[Resolution Feedback Store]
        G4[Continuous Learning / Prompt Updates]
    end

    S1 --> I1
    S2 --> I1
    S3 --> I2
    S4 --> I2
    S5 --> I3
    S6 --> I3
    S7 --> I4

    I1 --> A2
    I2 --> A1
    I3 --> A2
    I4 --> A4

    M1 --> A1
    M2 --> A1
    M3 --> A4
    M4 --> A3
    M4 --> A4

    A1 --> A2
    A2 --> A3
    A3 --> A4
    A4 --> A5

    A5 --> G1
    G1 --> R1
    G1 --> R2
    G1 --> R3
    G1 --> R4
    G1 --> R5

    R1 --> G2
    R2 --> G2
    R3 --> G2
    R4 --> G2
    R5 --> G2

    G2 --> G3
    G3 --> G4
    G4 --> M4