import os
import json
import csv
import pandas as pd

def get_folders(dataset):
    root_dir = os.path.join(r"C:\Users\thiag\DatasetsCv2", dataset)

    output_file = "paths.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        for filename in os.listdir(root_dir):
            path = os.path.join(root_dir, filename)
            if os.path.isdir(path):
                f.write(path + "\n")

    print(f"Paths salvos em {output_file}")
    
def get_annotations(): 
    input_file = r"C:\Users\thiag\Documents\Faculdade\TCC\TCC-Inferencia-de-Tom-de-Pele\files\CasualConversationsV2.json"
    output_file = r"C:\Users\thiag\Documents\Faculdade\TCC\TCC-Inferencia-de-Tom-de-Pele\files\ccv2_skin_tones.csv"

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subject_id", "fitzpatrick_type", "fitzpatrick_confidence", "monk_scale", "monk_confidence"])
        
        for item in data:
            subject_id = item.get("subject_id", "")
            fitz_type = item.get("fitzpatrick_skin_tone", {}).get("type", "")
            fitz_conf = item.get("fitzpatrick_skin_tone", {}).get("confidence", "")
            monk_scale = item.get("monk_skin_tone", {}).get("scale", "")
            monk_conf = item.get("monk_skin_tone", {}).get("confidence", "")
            
            writer.writerow([subject_id, fitz_type, fitz_conf, monk_scale, monk_conf])

def get_label():
    input_file = r"C:\Users\thiag\Documents\Faculdade\TCC\TCC-Inferencia-de-Tom-de-Pele\files\ccv2_skin_tones.csv"
    df = pd.read_csv(input_file, header=None)

    df.columns = ['subject_id','fitzpatrick_type','fitzpatrick_confidence','monk_scale','monk_confidence']

    result = (
        df.groupby("subject_id")[["fitzpatrick_type", "monk_scale"]]
        .agg(lambda x: x.mode().iat[0] if not x.mode().empty else None)
        .reset_index()
    )

    result.to_csv("ccv2_filtered.csv", index=False, encoding="utf-8")

    print(result.head())
