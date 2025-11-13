import pandas as pd
import matplotlib as plt
from sklearn import Con
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def get_confusion_matrix(true_file, pred_file):
    df_true = pd.read_csv(true_file)
    df_pred = pd.read_csv(pred_file)

    # Renomeia as colunas para padronizar
    df_true.columns = ["subject_id", "true_label"]
    df_pred.columns = ["subject_id", "pred_label"]

    # Junta os dois DataFrames pelo subject_id
    df = pd.merge(df_true, df_pred, on="subject_id", how="inner")

    # Remove linhas com valores ausentes (caso algum subject_id não tenha previsão)
    df = df.dropna(subset=["true_label", "pred_label"])

    # Gera a matriz de confusão
    cm = confusion_matrix(df["true_label"], df["pred_label"], labels=sorted(df["true_label"].unique()))

    # Exibe a matriz
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(df["true_label"].unique()))
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Matriz de Confusão CASCo na escala de Fitzpatrick")
    plt.show()