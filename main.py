from src.casco import run_casco
from src.results_analysis import get_confusion_matrix, get_matrix_file, get_class_distribution

run_casco(1)

true_file = r"C:\Users\thiag\Documents\Faculdade\TCC\TCC-Inferencia-de-Tom-de-Pele\files\ccv2_filtered.csv"
pred_file = r"C:\Users\thiag\Documents\Faculdade\TCC\TCC-Inferencia-de-Tom-de-Pele\results\casco\results_casco_1.csv"

get_matrix_file(true_file, pred_file, 'fitzpatrick_type')

matrix =  r"C:\Users\thiag\Documents\Faculdade\TCC\TCC-Inferencia-de-Tom-de-Pele\dados_para_matriz_confusao.csv"
get_confusion_matrix(matrix)

get_class_distribution('monk_scale')