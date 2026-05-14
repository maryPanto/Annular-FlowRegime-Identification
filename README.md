# Annular-FlowRegime-Identification

**Overview**

This project implements a machine learning pipeline for identifying two phase annular flow sub-regimes in circular pipes. Utilizing experimental data obtained with a DCCP-2 conductivity probe, the system extracts physics-informed features from time-series data and identifies to which annular flow sub-regime class each signal belongs.

**Technical Architecture**

Feature Extraction: Extracts volumetric, morphological and frequency-domain descriptors.

Condition-Based Partitioning: Implements an 85/15 split based on unique experimental runs to prevent data leakage and ensure model generalization.

System Reliability: Employs a Majority Voting strategy, aggregating segment-level predictions into a single, high-confidence run-level classification.

**How to Run**

Initialize Environment: pip install -r requirements.txt

Process and Partition: python generate_partition.py

Train and Evaluate: python main.py
