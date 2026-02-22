import pandas as pd
import matplotlib as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

def get_confusion_matrix(file):
    df = pd.read_csv(file)
    df["true_label"] = df["true_label"].astype(str)
    df["predicted_label"] = df["predicted_label"].astype(str)

    # fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(df["true_label"], df["predicted_label"], labels=sorted(df["true_label"].unique()), normalize='true').round(2)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(df["true_label"].unique()))
    # disp.plot(cmap="Blues", xticks_rotation=45, ax=ax, values_format='.2f')
    
    disp.plot(cmap="Blues", xticks_rotation=45)

    plt.title("Matriz de Confusão CASCo na escala Fitzpatrick")
    plt.show()

def load_true_labels(true_file):
    df_true = pd.read_csv(true_file)
    df_true.columns = ["subject_id", "true_label"]
    return df_true

def extract_subject_id(file_name):
    return file_name.split("_")[0]

def load_predictions(pred_file, pred_label_col):
    df_pred = pd.read_csv(pred_file)

    df_pred["subject_id"] = df_pred["file"].apply(extract_subject_id)

    df_pred = df_pred[["subject_id", pred_label_col]]
    df_pred.columns = ["subject_id", "pred_label"]

    return df_pred

def make_confusion_matrix(true_file, pred_file, pred_label_col, title):
    df_true = load_true_labels(true_file)
    df_pred = load_predictions(pred_file, pred_label_col)

    df = pd.merge(df_true, df_pred, on="subject_id", how="inner")
    df = df.dropna(subset=["true_label", "pred_label"])
    labels = sorted(df["true_label"].unique())

    cm = confusion_matrix(df["true_label"], df["pred_label"], labels=labels, normalize='all')

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", xticks_rotation=45)

    plt.title(title)
    plt.tight_layout()
    plt.show()

def get_matrix_file(true_file, pred_file, scale):
    OUTPUT_FILE = 'dados_para_matriz_confusao.csv'

    try:
        df_true = pd.read_csv(true_file, dtype={'subject_id': str})
        df_pred = pd.read_csv(pred_file)
        
    except FileNotFoundError as e:
        print(f"Erro: Arquivo não encontrado: {e.filename}")
        return

    df_true['subject_id'] = df_true['subject_id'].str.strip()
    df_pred['subject_id'] = df_pred['file'].str[:4].str.strip()

    df_final = pd.merge(
        df_pred, 
        df_true, 
        on='subject_id', 
        how='left' 
    )

    colunas_necessarias = ['subject_id', scale, 'tone label', 'file']
    
    colunas_presentes = [c for c in colunas_necessarias if c in df_final.columns]
    df_matriz = df_final[colunas_presentes].copy()

    df_matriz.rename(columns={
        scale: 'true_label',
        'tone label': 'predicted_label'
    }, inplace=True)

    df_matriz.to_csv(OUTPUT_FILE, index=False)

    print("-" * 50)
    print(f"✔️ Processamento concluído!")
    
    nulos = df_matriz['true_label'].isna().sum()
    if nulos > 0:
        print(f"⚠️ ATENÇÃO: {nulos} linhas ficaram sem true_label (não houve match).")
        print(f"Exemplo de subject_id que falhou: {df_pred.iloc[0]['subject_id']}")
    
    print(f"Total de linhas: {len(df_matriz)}")
    print("-" * 50)
    
def get_class_distribution(column):
    file = r'C:\Users\thiag\Documents\Faculdade\TCC\TCC-Inferencia-de-Tom-de-Pele\files\ccv2_filtered.csv'
    
    df = pd.read_csv(file)
    
    order = sorted(df[column].dropna().unique())
    ax = sns.countplot(data=df, x=column, order=order, palette="viridis")
    
    plt.title(f'Distribuição de Ocorrências: {column}', fontsize=15)
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Quantidade', fontsize=12)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()