# owasp-sc-token-
Token‑level CodeBERT model for localizing OWASP SC01/SC05/SC08 vulnerabilities in Solidity smart contracts using the SmartBugs‑Curated dataset

This project fine‑tunes microsoft/codebert-base for token‑level localization of smart contract vulnerabilities mapped to OWASP SC01 (Access Control), 
SC05 (Reentrancy), and SC08 (Arithmetic and Time Manipulation). 
The model is trained on the SmartBugs‑Curated dataset of 69 manually annotated Solidity contracts, using class‑weighted loss and oversampling to address strong label imbalance.


smartbugs-curated/: Original Solidity contracts and vulnerability annotations (external dataset).
​

data/first_dataset.json: Token‑level dataset derived from SmartBugs‑Curated.

generate_dataset.py: Script to build first_dataset.json from SmartBugs‑Curated.

train_codebert_sc05.py: Training script (baseline, class‑weighted, and oversampling, with configurable random seed).
​

predict_owasp_sc.py: Use the trained model to label tokens in new Solidity contracts.

results/: JSON files with evaluation metrics for each experiment (baseline, class‑weighted, oversampling, seeds 42/1/7)

# 1. Create environment and install dependencies
pip install -r requirements.txt  # or list transformers, datasets, torch, etc.

# 2. Generate token-level dataset (optional if data/first_dataset.json already exists)
python generate_dataset.py

# 3. Train best model (class-weighted + SC01 oversampling, seed=42)
python train_codebert_sc05.py

# 4. Run prediction on a contract
python predict_owasp_sc.py --model_dir codebert-sc05-model --input_file path/to/contract.sol
