import pandas as pd
import matplotlib as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

TRUE_FILE = r'C:\Users\thiag\Documents\Faculdade\TCC\TCC-Inferencia-de-Tom-de-Pele\files\ccv2\ccv2_filtered.csv'

def calculate_error(pred, true):
    ROMAN_SCALE = {
        'type i': 1, 'type ii': 2, 'type iii': 3, 
        'type iv': 4, 'type v': 5, 'type vi': 6
    }
    
    try:
        p_val = ROMAN_SCALE.get(str(pred).lower().strip())
        t_val = ROMAN_SCALE.get(str(true).lower().strip())
        
        if p_val is not None and t_val is not None:
            return abs(t_val - p_val)
        return None
    except:
        return None
    
def get_class_distribution(column):
    
    df = pd.read_csv(TRUE_FILE)
    
    order = sorted(df[column].dropna().unique())
    ax = sns.countplot(data=df, x=column, order=order, palette="viridis")
    
    plt.title(f'Distribuição de Ocorrências: {column}', fontsize=15)
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Quantidade', fontsize=12)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()    
    
def get_matrix_file(pred_file, scale, save_file):
    OUTPUT_FILE = 'confusion_matrix_file.csv'

    try:
        df_true = pd.read_csv(TRUE_FILE, dtype={'subject_id': str})
        df_pred = pd.read_csv(pred_file)
        
    except FileNotFoundError as e:
        print(f"Error: File not found {e.filename}")
        return

    df_true['subject_id'] = df_true['subject_id'].str.strip()
    df_pred['subject_id'] = df_pred['file'].str[:4].str.strip()

    df_final = pd.merge(
        df_pred, 
        df_true, 
        on='subject_id', 
        how='left' 
    )

    columns = ['subject_id', 'file', scale, 'tone label']
    
    selected_columns = [c for c in columns if c in df_final.columns]
    df_matrix = df_final[selected_columns].copy()

    df_matrix.rename(columns={
        scale: 'true_label',
        'tone label': 'predicted_label'
    }, inplace=True)
    
    df_matrix['error'] = df_matrix.apply(
        lambda row: calculate_error(row['predicted_label'], row['true_label']), 
        axis=1
    )

    if save_file:
        df_matrix.to_csv(OUTPUT_FILE, index=False)

    nulls = df_matrix['true_label'].isna().sum()
    if nulls > 0:
        print(f"{nulls} lines did not receive labels.")
    
    return df_matrix

def get_confusion_matrix(df_matrix, title):
    df_matrix["true_label"] = df_matrix["true_label"].astype(str)
    df_matrix["predicted_label"] = df_matrix["predicted_label"].astype(str)

    _, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(df_matrix["true_label"], df_matrix["predicted_label"], labels=sorted(df_matrix["true_label"].unique()), normalize='true').round(2)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(df_matrix["true_label"].unique()))
    disp.plot(cmap="Blues", xticks_rotation=45, ax=ax, values_format='.2f')

    plt.title(title)
    plt.show()
    
def analyse_results(predicions_file, method, scale, save_file=False):
    matrix_scale = 'monk_scale' if scale == 'monk' else 'fitzpatrick_type'
    title = f'Confusion Matrix {method} using {scale} scale'
    
    df_matrix = get_matrix_file(predicions_file, matrix_scale, save_file)
    get_confusion_matrix(df_matrix, title)