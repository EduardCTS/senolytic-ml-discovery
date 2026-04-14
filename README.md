# senolytic-ml-discovery
Initial data pipeline for BCL-xL (CHEMBL43) virtual screening. 

The goal here is to penalize physicochemical profiles linked to platelet toxicity before hitting the wet lab.

**Current build:**
- Pulls raw bioactivity data via ChEMBL API
- Generates Morgan Fingerprints (R=2, 2048 bits) for the GNN baseline
- Calculates basic ADME constraints (MolWt, LogP, TPSA) via RDKit to filter out toxic biodistributions early on

*WIP - Just laying the groundwork while the project scope is finalized.*
