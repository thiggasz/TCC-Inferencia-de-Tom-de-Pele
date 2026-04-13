import subprocess
import pandas as pd
import os
from tqdm import tqdm
from utils.utils import get_paths

def run_casco(n_dataset):
    bat_file = r"C:\Users\thiag\Documents\Faculdade\TCC\TCC-Inferencia-de-Tom-de-Pele\run_casco.bat"
    
    base_path = r"C:\Users\thiag\Dataset CCv2\Patchs"
    paths = os.listdir(base_path)
    
    casco_results = r"C:\Users\thiag\Documents\Faculdade\TCC\TCC-Inferencia-de-Tom-de-Pele\casco\result.csv"
    results_file = fr"C:\Users\thiag\Documents\Faculdade\TCC\TCC-Inferencia-de-Tom-de-Pele\results\casco\results_casco.csv"

    for path in tqdm(paths, desc="Calculating CASCo"):
        complete_path = os.path.join(base_path, path)
        print(f"Executando CASco em: {complete_path}")
        
        subprocess.run([bat_file, complete_path], shell=True)
        
        df = pd.read_csv(casco_results)
        df.to_csv(results_file, mode='a', index=False, header=False)