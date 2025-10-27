from datasets import load_dataset
import pandas as pd
import os

# Assurez-vous que les librairies sont installées (dans votre .venv)
# pip install datasets pandas pyarrow

print("Téléchargement du dataset Pirate (TeeZee/dolly-15k-pirate-speech)...")
# Charge le dataset
dataset = load_dataset("TeeZee/dolly-15k-pirate-speech", split="train")

# Convertit en Pandas
df = dataset.to_pandas()

# MODIFIÉ : On garde uniquement 'instruction' et 'response' pour simplifier
df = df[["instruction", "response"]]

# Sauvegarde en CSV
output_file = "dolly_pirate_simple_dataset.csv"
df.to_csv(output_file, index=False)

print(f"Dataset sauvegardé avec succès ! Fichier: {output_file}")
print(f"Nombre de paires: {len(df)}")
