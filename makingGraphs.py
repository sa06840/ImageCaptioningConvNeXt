import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plotDecoderLosses(transformerCsvPath, lstmCsvPath):
    """
    Plots the training and validation loss for Transformer and LSTM decoders
    on the same graph against epochs.
    Args:
        transformerCsvPath (str): Path to the CSV file for the Transformer decoder.
        lstmCsvPath (str): Path to the CSV file for the LSTM decoder.
    """
    transformerDf = pd.read_csv(transformerCsvPath)
    lstmDf = pd.read_csv(lstmCsvPath)

    # Create the plot
    plt.figure(figsize=(12, 7))

    # Plot Transformer Decoder losses
    plt.plot(transformerDf['epoch'], transformerDf['trainLoss'], label='Transformer Train Loss', color='blue', linestyle='-')
    plt.plot(transformerDf['epoch'], transformerDf['valLoss'], label='Transformer Val Loss', color='blue', linestyle='--')

    # Plot LSTM Decoder losses
    plt.plot(lstmDf['epoch'], lstmDf['trainLoss'], label='LSTM Train Loss', color='red', linestyle='-')
    plt.plot(lstmDf['epoch'], lstmDf['valLoss'], label='LSTM Val Loss', color='red', linestyle='--')

    # Add titles and labels
    plt.title('Training and Validation Loss Comparison: Transformer vs. LSTM Decoder (Flickr8k Dataset)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.show()
    plt.savefig('graphs/lossComparisonTransformerVsLstm.png')

def plotBleu4Scores(lstm_tf_csv_path, transformer_tf_csv_path, lstm_notf_csv_path, transformer_notf_csv_path):
    """
    Plots the BLEU-4 scores for four decoder training strategies on a single graph.

    Args:
        lstm_tf_csv_path (str): Path to the CSV file for the LSTM decoder with teacher forcing.
        transformer_tf_csv_path (str): Path to the CSV file for the Transformer decoder with teacher forcing.
        lstm_notf_csv_path (str): Path to the CSV file for the LSTM decoder without teacher forcing.
        transformer_notf_csv_path (str): Path to the CSV file for the Transformer decoder without teacher forcing.
    """
    # Load all four data files
    lstm_tf_df = pd.read_csv(lstm_tf_csv_path)
    transformer_tf_df = pd.read_csv(transformer_tf_csv_path)
    lstm_notf_df = pd.read_csv(lstm_notf_csv_path)
    transformer_notf_df = pd.read_csv(transformer_notf_csv_path)

    # --- Step 1: Data Manipulation ---
    # Add 1 to all epoch values
    lstm_tf_df['epoch'] += 1
    transformer_tf_df['epoch'] += 1
    lstm_notf_df['epoch'] += 1
    transformer_notf_df['epoch'] += 1
    
    # Multiply BLEU-4 scores by 100
    lstm_tf_df['bleu4'] *= 100
    transformer_tf_df['bleu4'] *= 100
    lstm_notf_df['bleu4'] *= 100
    transformer_notf_df['bleu4'] *= 100

    # Prepend a (0, 0) data point to each DataFrame
    df_epoch_0 = pd.DataFrame([{'epoch': 0, 'bleu4': 0.0}])
    lstm_tf_df = pd.concat([df_epoch_0, lstm_tf_df], ignore_index=True)
    transformer_tf_df = pd.concat([df_epoch_0, transformer_tf_df], ignore_index=True)
    lstm_notf_df = pd.concat([df_epoch_0, lstm_notf_df], ignore_index=True)
    transformer_notf_df = pd.concat([df_epoch_0, transformer_notf_df], ignore_index=True)

    # --- Step 2: Filter the 'without teacher forcing' data to stop at epoch 89 ---
    lstm_notf_df = lstm_notf_df[lstm_notf_df['epoch'] <= 90]
    transformer_notf_df = transformer_notf_df[transformer_notf_df['epoch'] <= 90]

    # --- Step 3: Create the plot with more detailed axes ---
    plt.figure(figsize=(10, 6))

    # Plot the four lines
    plt.plot(lstm_tf_df['epoch'], lstm_tf_df['bleu4'], label='LSTM + Att (TF)', color='blue', linestyle='-')
    plt.plot(transformer_tf_df['epoch'], transformer_tf_df['bleu4'], label='Transformer (TF)', color='green', linestyle='-')
    
    plt.plot(lstm_notf_df['epoch'], lstm_notf_df['bleu4'], label='LSTM + Att (No TF)', color='red', linestyle='--')
    plt.plot(transformer_notf_df['epoch'], transformer_notf_df['bleu4'], label='Transformer (No TF)', color='orange', linestyle='--')

    # Add titles and labels
    plt.title('BLEU-4 Score Comparison Across Decoder Architectures and Training Strategies', fontdict={'fontsize': 14, 'fontweight': 'bold'}, pad=20)
    plt.xlabel('Epoch', fontdict={'fontsize': 14}, labelpad=10)
    plt.ylabel('BLEU-4 Score', fontdict={'fontsize': 14}, labelpad=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=14)
    plt.tight_layout()
    
    # Set detailed x-axis ticks at intervals of 10
    max_epoch = max(lstm_tf_df['epoch'].max(), transformer_tf_df['epoch'].max())
    plt.xticks(np.arange(0, max_epoch + 1, 10), fontsize=12)
    
    # Set detailed y-axis ticks at intervals of 5
    plt.yticks(np.arange(0, 40, 5), fontsize=12)
   
    plt.savefig('graphs/bleuScoreComparisonTrainingStrategies.png', dpi=300)
    plt.show()

def get_best_bleu4_row(csv_path):
    """
    Reads a CSV file, finds the row with the highest 'bleu4' score, and returns it.

    Args:
        csv_path (str): The path to the CSV file containing the model's metrics.

    Returns:
        Dict[str, Any]: A dictionary representing the row with the highest BLEU-4 score.
        Returns an empty dictionary if the file is not found or 'bleu4' column is missing.
    """
    try:
        # Load the CSV data into a pandas DataFrame
        df = pd.read_csv(csv_path)

        # Check if the 'bleu4' column exists
        if 'bleu4' not in df.columns:
            print("Error: The 'bleu4' column was not found in the CSV file.")
            return {}
        
        # Find the index of the row with the highest 'bleu4' score
        # idxmax() returns the first index in case of ties
        best_row_index = df['bleu4'].idxmax()
        
        # Use .loc to retrieve the entire row as a Series, then convert to a dictionary
        best_row = df.loc[best_row_index].to_dict()
        
        return best_row
        
    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {}

def plotFinetunedBleu4Scores(
    no_finetune_csv,
    ft_5_7_1e4_20_csv,
    ft_5_7_1e5_40_csv,
    ft_5_7_1e6_40_csv,
    ft_3_7_1e4_20_csv,
    ft_1_7_1e6_40_csv,
    title,
    output_filename
):
  
    # plt.style.use('seaborn-v0_8-whitegrid')
    
    # Load all six data files
    df_list = [
        pd.read_csv(no_finetune_csv),
        pd.read_csv(ft_5_7_1e4_20_csv),
        pd.read_csv(ft_5_7_1e5_40_csv),
        pd.read_csv(ft_5_7_1e6_40_csv),
        pd.read_csv(ft_3_7_1e4_20_csv),
        pd.read_csv(ft_1_7_1e6_40_csv)
    ]
    
    # --- Step 1: Data Manipulation ---
    for df in df_list:
        # Add 1 to all epoch values to start from epoch 1
        df['epoch'] = df['epoch'] + 1
        # Multiply BLEU-4 scores by 100 for percentage
        df['bleu4'] *= 100
        # Prepend a (0, 0) data point to each DataFrame
        df_epoch_0 = pd.DataFrame([{'epoch': 0, 'bleu4': 0.0}])
        df = pd.concat([df_epoch_0, df], ignore_index=True)

    # --- Step 2: Plot the data ---
    plt.figure(figsize=(14, 8))

    # Define plot labels and styles
    labels = [
        'No Fine-tuning',
        'Layers 5-7, LR=1×10-4, Patience=20',
        'Layers 5-7, LR=1×10-5, Patience=40',
        'Layers 5-7, LR=1×10-6, Patience=40',
        'Layers 3-7, LR=1×10-4, Patience=20',
        'Layers 1-7, LR=1×10-6, Patience=40'
    ]
    colors = ['black', 'blue', 'green', 'orange', 'purple', 'red']
    linestyles = ['-', '-', '-', '--', '-', '--']
    
    for df, label, color, linestyle in zip(df_list, labels, colors, linestyles):
        plt.plot(df['epoch'], df['bleu4'], label=label, color=color, linestyle=linestyle, linewidth=2)

    # --- Step 3: Add titles, labels, and detailed axes ---
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontsize=14, labelpad=15)
    plt.ylabel('BLEU-4 Score', fontsize=14, labelpad=15)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, loc='upper left')
    plt.tight_layout()
    
    # Set detailed x-axis ticks at intervals of 10
    max_epoch = max(df['epoch'].max() for df in df_list)
    plt.xticks(np.arange(0, max_epoch + 1, 10), fontsize=12)
    
    # Set detailed y-axis ticks
    plt.yticks(np.arange(25, 40, 1), fontsize=12)

    # Ensure the 'graphs' directory exists before saving
    plt.savefig(output_filename, dpi=300)
    plt.show()


lstmMetricsTF = 'results/mscoco/17-07-2025(trainingTF-inferenceNoTF-noFinetuning)/metrics-lstmDecoder(trainingTF-inferenceNoTF-noFinetuning).csv'
transformerMetricsTF = 'results/mscoco/17-07-2025(trainingTF-inferenceNoTF-noFinetuning)/metrics-transformerDecoder(trainingTF-inferenceNoTF-noFinetuning).csv'
lstmMetricsNoTF = 'results/mscoco/20-07-2025(trainingNoTF-inferenceNoTF-noFinetuning)/metrics-lstmDecoder(trainingNoTF-inferenceNoTF-noFinetuning).csv'
transformerMetricsNoTF = 'results/mscoco/20-07-2025(trainingNoTF-inferenceNoTF-noFinetuning)/metrics-transformerDecoder(trainingNoTF-inferenceNoTF-noFinetuning).csv'
# plotDecoderLosses(transformerMetricsTF, lstmMetricsTF, lstmMetricsNoTF, transformerMetricsNoTF)
# plotBleu4Scores(lstmMetricsTF, transformerMetricsTF, lstmMetricsNoTF, transformerMetricsNoTF)

# path = 'results/mscoco/01-08-2025(trainingTF-inferenceNoTF-Finetuning5-lr1e6-40epochs)/metrics-transformerDecoder(trainingTF-inferenceNoTF-Finetuning5-1e-06).csv'
path = 'results/mscoco/12-08-2025(trainingTF-inferenceNoTF-Finetuning1-lr1e6-40epochs)/metrics-transformerDecoder(trainingTF-inferenceNoTF-Finetuning1-1e-06).csv'
# print(get_best_bleu4_row(path))


noFinetuned = 'results/mscoco/17-07-2025(trainingTF-inferenceNoTF-noFinetuning)/metrics-transformerDecoder(trainingTF-inferenceNoTF-noFinetuning).csv'
fineTuned1 = 'results/mscoco/24-07-2025(trainingTF-inferenceNoTF-Finetuning5-lr1e4)/metrics-transformerDecoder(trainingTF-inferenceNoTF-Finetuning5).csv'
fineTuned2 = 'results/mscoco/28-07-2025(trainingTF-inferenceNoTF-Finetuning5-lr1e5-40epochs)/metrics-transformerDecoder(trainingTF-inferenceNoTF-Finetuning5-1e-05).csv'
fineTuned3 = 'results/mscoco/01-08-2025(trainingTF-inferenceNoTF-Finetuning5-lr1e6-40epochs)/metrics-transformerDecoder(trainingTF-inferenceNoTF-Finetuning5-1e-06).csv'
fineTuned4 = 'results/mscoco/24-07-2025(trainingTF-inferenceNoTF-Finetuning3-lr1e4)/metrics-transformerDecoder(trainingTF-inferenceNoTF-Finetuning3).csv'
fineTuned5 = 'results/mscoco/12-08-2025(trainingTF-inferenceNoTF-Finetuning1-lr1e6-40epochs)/metrics-transformerDecoder(trainingTF-inferenceNoTF-Finetuning1-1e-06).csv'
plotFinetunedBleu4Scores(
    noFinetuned,
    fineTuned1,
    fineTuned2,
    fineTuned3,
    fineTuned4,
    fineTuned5,
    title='BLEU-4 Score Comparison for Transformer Decoder with Finetuning ConvNeXt',
    output_filename='graphs/bleuScoreComparisonFinetuning.png'
)