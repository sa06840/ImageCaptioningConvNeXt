import pandas as pd
import matplotlib.pyplot as plt


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


def plotBleu4Scores(transformerCsvPath, lstmCsvPath):
    """
    Plots the BLEU-4 scores for Transformer and LSTM decoders
    on the same graph against epochs.
    Args:
        transformerCsvPath (str): Path to the CSV file for the Transformer decoder.
        lstmCsvPath (str): Path to the CSV file for the LSTM decoder.
    """
    # Load the data
    transformerDf = pd.read_csv(transformerCsvPath)
    lstmDf = pd.read_csv(lstmCsvPath)

    # Create the plot
    plt.figure(figsize=(12, 7))
    # Plot Transformer Decoder BLEU-4 scores
    plt.plot(transformerDf['epoch'], transformerDf['bleu4'], label='Transformer BLEU-4', color='blue', linestyle='-')
    # Plot LSTM Decoder BLEU-4 scores
    plt.plot(lstmDf['epoch'], lstmDf['bleu4'], label='LSTM BLEU-4', color='red', linestyle='-')
    # Add titles and labels
    plt.title('BLEU-4 Score Comparison: Transformer vs. LSTM Decoder (Flickr8k Dataset)')
    plt.xlabel('Epoch')
    plt.ylabel('BLEU-4 Score')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.show()


lstmMetrics = 'results/flickr8k/29-6-2025(main)/metrics-lstmDecoder.csv'
transformerMetrics = 'results/flickr8k/29-6-2025(main)/metrics-transformerDecoder.csv'
# plotDecoderLosses(transformerMetrics, lstmMetrics)
plotBleu4Scores(transformerMetrics, lstmMetrics)
