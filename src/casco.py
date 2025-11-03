import subprocess
from json import dumps
import stone
from pathlib import Path
import pandas as pd

def run_casco():
    bat_file = r"C:\Users\thiag\OneDrive\Documentos\Faculdade\TCC\TCC-Inferencia-de-Tom-de-Pele\run_casco.bat"
    paths_file = r"C:\Users\thiag\OneDrive\Documentos\Faculdade\TCC\TCC-Inferencia-de-Tom-de-Pele\paths.txt"
    
    casco_results = r"C:\Users\thiag\OneDrive\Documentos\Faculdade\TCC\TCC-Inferencia-de-Tom-de-Pele\casco\result.csv"
    results_file = r"C:\Users\thiag\OneDrive\Documentos\Faculdade\TCC\TCC-Inferencia-de-Tom-de-Pele\results\results_casco_3.csv"

    with open(paths_file, "r", encoding="utf-8") as f:
        paths = [line.strip() for line in f if line.strip()]

    for path in paths:
        print(f"Executando CASco em: {path}")
        subprocess.run([bat_file, path], shell=True)
        
        df = pd.read_csv(casco_results)
        df.to_csv(results_file, mode='a', index=False, header=False)
        
        
        
        
        
        

