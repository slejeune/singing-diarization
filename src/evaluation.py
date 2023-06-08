from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.database.util import load_rttm

import matplotlib.pyplot as plt
import pandas

class Evaluation:
    
    def __init__(self) -> None:
        self.der_metric = DiarizationErrorRate()
    
    def evaluate(self, reference_path, hypothesis_path):
        '''
        Computes the Diarization Error Rate.
        
        Args:
            reference: the ground truth
            hypothesis: the predicted labels
        Returns:
            int: diarization error rate
        '''
        file_name = reference_path.split("/")[-1].split(".")[0]
        reference = load_rttm(reference_path)[file_name]
        hypothesis = load_rttm(hypothesis_path)[file_name]
        
        return self.der_metric(reference, hypothesis)
    
    def accumulated(self):
        """
        Computes the accumulated value across all results.
        
        Returns:
            int: accumulated value
        """
        return abs(self.der_metric)
    
    def report(self):
        """
        Provides a summary of all files evaluated.
        
        Returns:
            pandas.core.frame.DataFrame: summary of all results
        """
        return self.der_metric.report(display=True)
        
    def plot(self, report:pandas.core.frame.DataFrame):
        """
        Creates and saves a graph of the results.
        
        Args:
            report: dataframe of the evaluation
            output_path: location where the graph will be saved
        """
        
        fig, ax = plt.subplots(1, 2, figsize=(12,8))
        
        # Speaking
        speaking_df = report[:int((report.shape[0]-1)/2)]
        speaking_df.sort_index(inplace=True)
        ax[0].bar(speaking_df.index, speaking_df.iloc[:, 0].tolist(), color="cornflowerblue")
        ax[0].set_title("Speaking")
        ax[0].tick_params(labelrotation=90)
        ax[0].set_ylabel("DER")
        ax[0].set_ylim([0,100])
        
        # Singing
        singing_df = report[int((report.shape[0]-1)/2):-1]
        singing_df.sort_index(inplace=True)
        ax[1].bar(singing_df.index, singing_df.iloc[:, 0].tolist(), color="goldenrod")
        ax[1].set_title("Singing")
        ax[1].tick_params(labelrotation=90)
        ax[1].set_ylabel("DER")
        ax[1].set_ylim([0,100])
        
        plt.tight_layout()
        plt.show()