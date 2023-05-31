from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.database.util import load_rttm

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
            <type?>: summary of all results
        """
        return self.der_metric.report(display=True)
        
    def visualize_heatmap(self, output_path):
        """
        Creates and saves a heatmap of the results.
        
        Args:
            output_path: location where the graph will be saved
        """
        pass    
    