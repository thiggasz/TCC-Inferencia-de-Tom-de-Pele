import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import numpy as np
from analysis.results_analysis import get_matrix_file

sns.set_theme(style="whitegrid", palette="muted")

TRUE_FILE = r"C:\Users\thiag\Documents\Faculdade\TCC\TCC-Inferencia-de-Tom-de-Pele\files\ccv2\ccv2_filtered.csv"
ANALYSIS_FILE = r"C:\Users\thiag\Documents\Faculdade\TCC\TCC-Inferencia-de-Tom-de-Pele\results\ccv2_analysis.csv"

def plot_correlation_heatmap(df, interest_columns):
    plt.figure(figsize=(10, 8))
    
    df_filtered = df[interest_columns]
    
    corr_matrix = df_filtered.corr(method='spearman')
    
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    
    plt.title("Correlation with error Heatmap", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_dispersion_tendency(df, attribute_column, error_column="error"):
    plt.figure(figsize=(8, 6))
    
    sns.regplot(
        data=df, 
        x=attribute_column, 
        y=error_column, 
        scatter_kws={'alpha':0.3, 's':20}, 
        line_kws={'color':'red', 'linewidth':2},
        order=2 
    )
    
    plt.title(f"Impact of {attribute_column} on the prediction error", fontsize=14)
    plt.xlabel(attribute_column)
    plt.ylabel(error_column)
    plt.tight_layout()
    plt.show()

def plot_error_boxplot(df, attribute_column, error_column='error'):
    plt.close('all')
    plt.figure(figsize=(10, 6))
    
    sns.boxplot(
        data=df, 
        x=error_column, 
        y=attribute_column,
        hue=error_column, 
        palette="Reds",
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 6},
        legend=False
    )
    
    plt.title(f'Distribution of {attribute_column} by error magnitude', fontsize=14, fontweight='bold')
    plt.xlabel('Error magnitude (Distância Absoluta)', fontsize=12)
    plt.ylabel(attribute_column.capitalize(), fontsize=12)
    
    plt.tight_layout()
    plt.show()

def plot_predictions_range(df, attribute_column):
    plt.figure(figsize=(10, 6))
    
    unique_errors = sorted(df['error'].dropna().unique())
    colors = ['#2ca02c', '#98df8a', '#ffbb78', '#ff7f0e', '#d62728', '#8c564b']
    used_colors = colors[:len(unique_errors)]
    
    ax = sns.histplot(
        data=df, 
        x=attribute_column, 
        hue='error', 
        multiple='fill', 
        bins='auto', 
        palette=used_colors, 
        edgecolor='black', 
        alpha=0.85
    )

    plt.title(f'Proportional error distribution by {attribute_column}', fontsize=14)
    plt.xlabel(attribute_column.capitalize(), fontsize=12)
    plt.ylabel('Images percentage (%)', fontsize=12)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.05, 1), title='Error level')
    plt.tight_layout()
    plt.show()
   
def plot_quantitative_distribution(df, attribute_column):
    plt.figure(figsize=(10, 6))

    ax = sns.histplot(
        data=df, 
        x=attribute_column, 
        bins='auto', 
        color='steelblue', 
        edgecolor='black'
    )

    total_imgs = df[attribute_column].notna().sum()

    for p in ax.patches:
        height = p.get_height()
        if pd.isna(height) or height == 0:
            continue
        
        
    plt.title(f'Number of images distribution for {attribute_column}', fontsize=14, fontweight='bold')
    plt.xlabel(attribute_column.capitalize(), fontsize=12)
    plt.ylabel('Number of images', fontsize=12)
    plt.ylim(0, ax.get_ylim()[1] * 1.15)
    plt.tight_layout()
    plt.show()
   
def plot_error_bias(df, true_label_column='true_label', error_column='error'):
    df_grouped = df.groupby([true_label_column, error_column]).size().unstack(fill_value=0)
    
    df_percentage = df_grouped.div(df_grouped.sum(axis=1), axis=0) * 100
    
    colors = ['#2ca02c', '#ffbb78', '#ff7f0e', '#d62728', '#9467bd', '#8c564b']
    used_colors = colors[:len(df_percentage.columns)]
    
    ax = df_percentage.plot(kind='bar', stacked=True, figsize=(10, 6), 
                            color=used_colors, edgecolor='black', alpha=0.85)

    plt.title('Error distribution per True Class (Bias Analysis)', fontsize=14, fontweight='bold')
    plt.xlabel('True Label (Skin Tone)', fontsize=12)
    plt.ylabel('Images Percentage (%)', fontsize=12)
    plt.legend(title='Error Magnitude', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0) 
    plt.tight_layout()
    plt.show()   

def create_dataframe(df_pred, df_analysis, output='prediction_analysis.csv', save_file=False):
    df_final = pd.merge(
        df_pred, 
        df_analysis, 
        on='file', 
        how='left' 
    )

    if save_file:
        df_final.to_csv(output, index=False)
    
    nulls = df_final['true_label'].isna().sum()
    if nulls > 0:
        print(f"{nulls} lines did not receive labels.")
        
    return df_final
    
if __name__ == "__main__":
    PRED_FILE = r'C:\Users\thiag\Documents\Faculdade\TCC\TCC-Inferencia-de-Tom-de-Pele\results_ita.csv'
    class_filter = None
    
    df_matrix = get_matrix_file(PRED_FILE, 'fitzpatrick_type', False)
    df_analysis = pd.read_csv(ANALYSIS_FILE, dtype={'subject_id': str})
    
    df = create_dataframe(df_matrix, df_analysis)
    
    if class_filter:
        df= df[df['true_label'] == class_filter]
        print(f"Analysing class: {class_filter}. Total images: {len(df)}")
    
    columns = ['error', 'luminance mean', 'luminance std', 'noise', 'sharpness', 'yaw', 'pitch', 'roll', 'contrast', 'spill', 'temperature']
    
    plot_correlation_heatmap(df, columns)
    plot_error_bias(df)
    
    columns_to_analyse = ['luminance mean', 'luminance std', 'spill', 'temperature']

    for column in columns_to_analyse:
        plot_dispersion_tendency(df, attribute_column=column)
        plot_error_boxplot(df, attribute_column=column)
        plot_predictions_range(df, attribute_column=column)
        plot_quantitative_distribution(df, attribute_column=column)
        