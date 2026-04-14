"""
BCL-xL Senolytic Dataset Builder v0.1
Author: Eduardo T.
Objective: Extract BCL-xL (Target: CHEMBL43) bioactivity data, generate Morgan Fingerprints, 
and calculate basic physicochemical properties for initial ML screening and toxicity filtering.
"""

from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import pandas as pd

def fetch_bclxl_data():
    print("Fetching BCL-xL bioactivity data from ChEMBL...")
    # CHEMBL43 is the target ID for Human BCL-xL (Bcl-2-like protein 1)
    activities = new_client.activity.filter(target_chembl_id='CHEMBL43', pchembl_value__isnull=False)
    
    data = []
    for act in activities[:500]: # Fetching first 500 for the skeleton/proof-of-concept
        data.append({
            'molecule_chembl_id': act['molecule_chembl_id'],
            'smiles': act['canonical_smiles'],
            'pIC50': float(act['pchembl_value'])
        })
    return pd.DataFrame(data)

def generate_features(df):
    print("Generating Morgan Fingerprints and Physicochemical descriptors...")
    features = []
    for index, row in df.iterrows():
        mol = Chem.MolFromSmiles(row['smiles'])
        if mol is not None:
            # Morgan Fingerprint (Radius 2, 2048 bits) for Graph Neural Network baseline
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            
            # Key features for Platelet/Biodistribution Toxicity Risk
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            
            features.append({'mw': mw, 'logp': logp, 'tpsa': tpsa, 'valid': True})
        else:
            features.append({'mw': None, 'logp': None, 'tpsa': None, 'valid': False})
            
    feature_df = pd.DataFrame(features)
    return pd.concat([df, feature_df], axis=1).dropna()

if __name__ == "__main__":
    bclxl_df = fetch_bclxl_data()
    processed_df = generate_features(bclxl_df)
    
    print(f"Dataset built successfully. Shape: {processed_df.shape}")
    processed_df.to_csv("bclxl_initial_dataset.csv", index=False)
    print("Saved to bclxl_initial_dataset.csv. Ready for ML baseline training.")