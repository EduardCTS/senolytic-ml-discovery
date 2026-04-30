# senolytic-ml-discovery
Initial data pipeline for BCL-xL (CHEMBL43) virtual screening. The main goal here is to penalize physicochemical profiles linked to platelet toxicity before any of this hits a wet lab. 

For this current build I basically wrote a script that pulls the raw bioactivity data directly via the ChEMBL API. It then generates the Morgan Fingerprints (R=2, 2048 bits) to serve as our GNN baseline, while also calculating the basic ADME constraints like MolWt, LogP, and TPSA via RDKit so we can filter out toxic biodistributions right out of the gate.

WIP. Just laying the groundwork in code while the actual project scope gets finalized.

Update: Pushed a new module (admet_pains_filter.py) to handle the initial data scrubbing, aggressively filtering out PAINS and enforcing strict Lipinski boundaries right away to avoid feeding garbage pharmacokinetic profiles to the graph neural network during training
