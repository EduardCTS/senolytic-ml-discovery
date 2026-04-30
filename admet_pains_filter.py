"""
ADMET & PAINS Filter v0.1
Author: Eduardo T.
Objective: Aggressively filter the initial BCL-xL dataset to remove Pan Assay Interference Compounds (PAINS) 
and apply strict Lipinski/Veber rules to ensure baseline pharmacokinetic viability before ML training.
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import FilterCatalog

def load_data(filepath="bclxl_initial_dataset.csv"):
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        print("Initial dataset not found. Run builder first.")
        return None

def apply_strict_filters(df):
    print("Initializing PAINS catalog and ADMET screening...")
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog.FilterCatalog(params)
    
    clean_data = []
    
    for index, row in df.iterrows():
        mol = Chem.MolFromSmiles(row['smiles'])
        if mol is None:
            continue
            
        # PAINS Alert check
        if catalog.HasMatch(mol):
            continue
            
        # Strict Lipinski/Veber boundary checks for baseline bioavailability
        # Penalizing massive, greasy molecules that tend to accumulate non-specifically
        if row['mw'] > 500 or row['logp'] > 5.0 or row['tpsa'] > 140:
            continue
            
        clean_data.append(row)
        
    filtered_df = pd.DataFrame(clean_data)
    print(f"Filtering complete. Retained {len(filtered_df)} high-quality leads from original {len(df)}.")
    return filtered_df

if __name__ == "__main__":
    raw_df = load_data()
    if raw_df is not None:
        clean_df = apply_strict_filters(raw_df)
        clean_df.to_csv("bclxl_cleaned_training_set.csv", index=False)
        print("Saved scrubbed dataset. Ready for GNN ingestion.")
