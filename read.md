# 🚀 Campaign Optimization System

AI-driven campaign optimization system combining **LightGBM predictions, causal modeling, and LLM-powered recommendations (CrewAI)** with an interactive **Streamlit UI** for ROI-based customer targeting and marketing decisioning.

---

## 🔥 Key Features

- 📊 ROI-driven campaign optimization (not just prediction)
- 🤖 LightGBM-based customer behavior modeling
- 🎯 Causal modeling for uplift & spend estimation
- 🧠 LLM-powered recommendation engine
- 🤝 Agentic decision system (CrewAI)
- 📈 End-to-end pipeline from data → decision → business output
- ⚙️ Production-oriented with MLflow monitoring

---

## 🧠 System Architecture Overview

This system is designed as a **multi-layered intelligent pipeline** that transforms raw customer data into actionable marketing decisions.

---

## 📊 Architecture Diagram

```mermaid
flowchart TB

    subgraph Data_Layer[Data Layer]
        A1[Customer Data]
        A2[Purchase History]
        A3[Feedback / Reviews]
        A4[Product Catalog]
    end

    subgraph Modeling_Layer[Modeling Layer]
        B1[Causal Spend Model<br/>Training Pipeline]
        B2[Buy / No-Buy Prediction<br/>Batch Inference Pipeline]
        B21[Model Monitoring<br/>MLflow]
        B3[LLM Recommendation Engine]
        B4[Product Ranking + Textual Improvements]
    end

    subgraph Decision_Layer[Decision Layer]
        C1[Campaign Eligibility Logic]
        C2[Budget Threshold Rules]
        C3[Quarterly Profit Projection]
    end

    subgraph Business_Output[Business Output]
        D1[Upcoming Campaign Product List]
        D2[Expected Profit]
        D3[Go / No-Go Decision]
    end

    A1 --> B1
    A2 --> B1
    B1 --> B2
    B2 --> B21
    B21 --> B1

    A3 --> B3
    A4 --> B3
    B3 --> B4

    B2 --> C1
    B4 --> C1
    C1 --> C2
    C2 --> C3

    C1 --> D1
    C3 --> D2
    C2 --> D3
