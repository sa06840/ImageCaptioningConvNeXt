import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import operator 


# EDA

def visualizeWordFrequencies(baseDataPath, baseFilename, topN):
    wordFreqDict = {}
    wordMapPath = os.path.join(baseDataPath, 'WORDMAP_' + baseFilename + '.json')
    with open(wordMapPath, 'r') as j:
        wordMap = json.load(j)

    specialTokens = {wordMap['<start>'], wordMap['<end>'], wordMap['<pad>'], wordMap['<unk>']}
    stopWords = {'a', 'an', 'the', 'and', 'but', 'or', 'on', 'in', 'at', 'with', 'by', 'of', 'for', 'is', 'it', 'its', 'to',
        'from', 'as', 'that', 'this', 'he', 'she', 'his', 'her', 'we', 'our', 'they', 'their', 'be', 'are', 'was', 'were'}
    revWordMap = {v: k for k, v in wordMap.items()}

    for split in ['TRAIN', 'VAL', 'TEST']:
        captionsFilePath = os.path.join(baseDataPath, split + '_CAPTIONS_' + baseFilename + '.json')
        with open(captionsFilePath, 'r') as j:
            allCaptionsList = json.load(j)
            for captionIds in allCaptionsList:
                for wordId in captionIds:
                    wordString = revWordMap.get(wordId)
                    if wordId not in specialTokens and wordString and wordString not in stopWords:
                        wordFreqDict[wordId] = wordFreqDict.get(wordId, 0) + 1
  
    sortedWordFreq = sorted(wordFreqDict.items(), key=lambda item: item[1], reverse=True)
    topWordsIdsWithFreqs = sortedWordFreq[:topN]
    topWordsIds = [item[0] for item in topWordsIdsWithFreqs]
    topWordsFreqs = [item[1] for item in topWordsIdsWithFreqs]
    topWordsStrings = []
    for wordId in topWordsIds:
        wordString = revWordMap.get(wordId)
        topWordsStrings.append(wordString)

    plt.figure(figsize=(20, 10))
    bars = plt.barh(topWordsStrings[::-1], topWordsFreqs[::-1], color='steelblue', alpha=0.9)
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 50, bar.get_y() + bar.get_height()/2, f'{width}', va='center', fontsize=12)
    plt.title(f'Top {topN} Most Frequent Words in the Dataset (Excluding Stop Words)', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Frequency', fontsize=16, labelpad=15)
    plt.ylabel('Words', fontsize=16, labelpad=15)
    plt.xticks(fontsize=14, rotation=0) 
    plt.yticks(fontsize=14) 
    plt.tight_layout()
    plt.grid(axis='x', linestyle='--', alpha=0.6) 
    outputPath = 'graphs/EDA/wordFrequencies.png'
    plt.savefig(outputPath, dpi=300)
    plt.show()


def visualizeCaptionLengths(baseDataPath, baseFilename, numBins):
    allCaptionLengths = []
    for split in ['TRAIN', 'VAL', 'TEST']:
        caplensFilePath = os.path.join(baseDataPath, split + '_CAPLENS_' + baseFilename + '.json')
        with open(caplensFilePath, 'r') as j:
            captionLengthsList = json.load(j)
            allCaptionLengths.extend(captionLengthsList)

    lengthsArray = np.array(allCaptionLengths)
    
    plt.figure(figsize=(12, 7))
    plt.hist(lengthsArray, bins=numBins, color='steelblue', edgecolor='black', alpha=0.9)
    plt.title('Distribution of Caption Lengths in the Dataset', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Caption Length (including special tokens)', fontsize=14, labelpad=15)
    plt.ylabel('Frequency', fontsize=14, labelpad=15)
    meanLength = lengthsArray.mean()
    plt.axvline(meanLength, color='red', linestyle='--', linewidth=2, label=f'Mean Length: {meanLength:.2f}')
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    outputPath = 'graphs/EDA/captionLengths.png'
    plt.savefig(outputPath, dpi=300)
    plt.show()


# Results

def plotDecoderLosses(transformerCsvPath, lstmCsvPath):
    transformerDf = pd.read_csv(transformerCsvPath)
    lstmDf = pd.read_csv(lstmCsvPath)

    plt.figure(figsize=(12, 7))

    plt.plot(transformerDf['epoch'], transformerDf['trainLoss'], label='Transformer Train Loss', color='blue', linestyle='-')
    plt.plot(transformerDf['epoch'], transformerDf['valLoss'], label='Transformer Val Loss', color='blue', linestyle='--')
    plt.plot(lstmDf['epoch'], lstmDf['trainLoss'], label='LSTM Train Loss', color='red', linestyle='-')
    plt.plot(lstmDf['epoch'], lstmDf['valLoss'], label='LSTM Val Loss', color='red', linestyle='--')

    plt.title('Training and Validation Loss Comparison: Transformer vs. LSTM Decoder (Flickr8k Dataset)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig('graphs/lossComparisonTransformerVsLstm.png')


def plotBleu4Scores(lstm_tf_csv_path, transformer_tf_csv_path, lstm_notf_csv_path, transformer_notf_csv_path):
    lstm_tf_df = pd.read_csv(lstm_tf_csv_path)
    transformer_tf_df = pd.read_csv(transformer_tf_csv_path)
    lstm_notf_df = pd.read_csv(lstm_notf_csv_path)
    transformer_notf_df = pd.read_csv(transformer_notf_csv_path)

    lstm_tf_df['epoch'] += 1
    transformer_tf_df['epoch'] += 1
    lstm_notf_df['epoch'] += 1
    transformer_notf_df['epoch'] += 1
    
    lstm_tf_df['bleu4'] *= 100
    transformer_tf_df['bleu4'] *= 100
    lstm_notf_df['bleu4'] *= 100
    transformer_notf_df['bleu4'] *= 100

    df_epoch_0 = pd.DataFrame([{'epoch': 0, 'bleu4': 0.0}])
    lstm_tf_df = pd.concat([df_epoch_0, lstm_tf_df], ignore_index=True)
    transformer_tf_df = pd.concat([df_epoch_0, transformer_tf_df], ignore_index=True)
    lstm_notf_df = pd.concat([df_epoch_0, lstm_notf_df], ignore_index=True)
    transformer_notf_df = pd.concat([df_epoch_0, transformer_notf_df], ignore_index=True)

    lstm_notf_df = lstm_notf_df[lstm_notf_df['epoch'] <= 90]
    transformer_notf_df = transformer_notf_df[transformer_notf_df['epoch'] <= 90]

    plt.figure(figsize=(10, 6))

    plt.plot(lstm_tf_df['epoch'], lstm_tf_df['bleu4'], label='LSTM + Att (TF)', color='blue', linestyle='-')
    plt.plot(transformer_tf_df['epoch'], transformer_tf_df['bleu4'], label='Transformer (TF)', color='green', linestyle='-')
    plt.plot(lstm_notf_df['epoch'], lstm_notf_df['bleu4'], label='LSTM + Att (No TF)', color='red', linestyle='--')
    plt.plot(transformer_notf_df['epoch'], transformer_notf_df['bleu4'], label='Transformer (No TF)', color='orange', linestyle='--')

    plt.title('BLEU-4 Score Comparison Across Decoder Architectures and Training Strategies', fontdict={'fontsize': 14, 'fontweight': 'bold'}, pad=20)
    plt.xlabel('Epoch', fontdict={'fontsize': 14}, labelpad=10)
    plt.ylabel('BLEU-4 Score', fontdict={'fontsize': 14}, labelpad=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=14)
    plt.tight_layout()
    
    max_epoch = max(lstm_tf_df['epoch'].max(), transformer_tf_df['epoch'].max())
    plt.xticks(np.arange(0, max_epoch + 1, 10), fontsize=12)
    plt.yticks(np.arange(0, 40, 5), fontsize=12)
   
    plt.savefig('graphs/bleuScoreComparisonTrainingStrategies.png', dpi=300)
    plt.show()


def plotFinetunedBleu4Scores(no_finetune_csv, ft_5_7_1e4_20_csv, ft_5_7_1e5_40_csv, ft_5_7_1e6_40_csv, ft_3_7_1e4_20_csv, ft_1_7_1e6_40_csv,
    title, output_filename):

    df1 = pd.read_csv(no_finetune_csv)
    df2 = pd.read_csv(ft_5_7_1e4_20_csv)
    df3 = pd.read_csv(ft_5_7_1e5_40_csv)
    df4 = pd.read_csv(ft_5_7_1e6_40_csv)
    df5 = pd.read_csv(ft_3_7_1e4_20_csv)
    df6 = pd.read_csv(ft_1_7_1e6_40_csv)
  
    df_list = [df1, df2, df3, df4, df5, df6]
    
    for df in df_list:
        df['epoch'] = df['epoch'] + 1
        df['bleu4'] *= 100
        df_epoch_0 = pd.DataFrame([{'epoch': 0, 'bleu4': 0.0}])
        df = pd.concat([df_epoch_0, df], ignore_index=True)

    plt.figure(figsize=(14, 8))
    labels = [
        'No Fine-tuning',
        'Layers 5-7, LR=1$\\times 10^{-4}$, Patience=20',
        'Layers 5-7, LR=1$\\times 10^{-5}$, Patience=40',
        'Layers 5-7, LR=1$\\times 10^{-6}$, Patience=40',
        'Layers 3-7, LR=1$\\times 10^{-4}$, Patience=20',
        'Layers 1-7, LR=1$\\times 10^{-6}$, Patience=40'
    ]
    colors = ['black', 'blue', 'green', 'orange', 'purple', 'red']
    linestyles = ['-', '-', '-', '--', '-', '--']
    
    for df, label, color, linestyle in zip(df_list, labels, colors, linestyles):
        plt.plot(df['epoch'], df['bleu4'], label=label, color=color, linestyle=linestyle, linewidth=2)

    plt.title(title, fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontsize=16, labelpad=15)
    plt.ylabel('BLEU-4 Score', fontsize=16, labelpad=15)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, loc='upper left')
    plt.tight_layout()
    
    all_max_epochs = []
    for df in df_list:
        current_max_epoch = df['epoch'].max()
        all_max_epochs.append(current_max_epoch)
    max_epoch = max(all_max_epochs)
    plt.xticks(np.arange(0, max_epoch + 1, 10), fontsize=14)
    plt.yticks(np.arange(25, 40, 1), fontsize=14)
    plt.savefig(output_filename, dpi=300)
    plt.show()


# visualizeWordFrequencies('cocoDataset/inputFiles', 'coco_5_cap_per_img_5_min_word_freq', 20)
# visualizeCaptionLengths(baseDataPath='cocoDataset/inputFiles', baseFilename='coco_5_cap_per_img_5_min_word_freq', numBins=40)

lstmMetricsTF = 'results/mscoco/17-07-2025(trainingTF-inferenceNoTF-noFinetuning)/metrics-lstmDecoder(trainingTF-inferenceNoTF-noFinetuning).csv'
transformerMetricsTF = 'results/mscoco/17-07-2025(trainingTF-inferenceNoTF-noFinetuning)/metrics-transformerDecoder(trainingTF-inferenceNoTF-noFinetuning).csv'
lstmMetricsNoTF = 'results/mscoco/20-07-2025(trainingNoTF-inferenceNoTF-noFinetuning)/metrics-lstmDecoder(trainingNoTF-inferenceNoTF-noFinetuning).csv'
transformerMetricsNoTF = 'results/mscoco/20-07-2025(trainingNoTF-inferenceNoTF-noFinetuning)/metrics-transformerDecoder(trainingNoTF-inferenceNoTF-noFinetuning).csv'
# plotDecoderLosses(transformerMetricsTF, lstmMetricsTF, lstmMetricsNoTF, transformerMetricsNoTF)
# plotBleu4Scores(lstmMetricsTF, transformerMetricsTF, lstmMetricsNoTF, transformerMetricsNoTF)


noFinetuned = 'results/mscoco/03_17-07-2025(trainingTF-inferenceNoTF-noFinetuning)/metrics-transformerDecoder(trainingTF-inferenceNoTF-noFinetuning).csv'
fineTuned1 = 'results/mscoco/05_24-07-2025(trainingTF-inferenceNoTF-Finetuning5-lr1e4)/metrics-transformerDecoder(trainingTF-inferenceNoTF-Finetuning5).csv'
fineTuned2 = 'results/mscoco/07_28-07-2025(trainingTF-inferenceNoTF-Finetuning5-lr1e5-40epochs)/metrics-transformerDecoder(trainingTF-inferenceNoTF-Finetuning5-1e-05).csv'
fineTuned3 = 'results/mscoco/08_01-08-2025(trainingTF-inferenceNoTF-Finetuning5-lr1e6-40epochs)/metrics-transformerDecoder(trainingTF-inferenceNoTF-Finetuning5-1e-06).csv'
fineTuned4 = 'results/mscoco/06_24-07-2025(trainingTF-inferenceNoTF-Finetuning3-lr1e4)/metrics-transformerDecoder(trainingTF-inferenceNoTF-Finetuning3).csv'
fineTuned5 = 'results/mscoco/09_12-08-2025(trainingTF-inferenceNoTF-Finetuning1-lr1e6-40epochs)/metrics-transformerDecoder(trainingTF-inferenceNoTF-Finetuning1-1e6).csv'
plotFinetunedBleu4Scores(
    noFinetuned,
    fineTuned1,
    fineTuned2,
    fineTuned3,
    fineTuned4,
    fineTuned5,
    title='BLEU-4 Score Comparison for Transformer Decoder with Finetuning ConvNeXt',
    output_filename='graphs/resultsGraphs/bleuScoreComparisonFinetuning.png'
)
