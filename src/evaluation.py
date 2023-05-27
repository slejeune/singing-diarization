from pyannote.metrics.diarization import DiarizationErrorRate

class Evaluation:
    
    def __init__(self) -> None:
        self.der_metric = DiarizationErrorRate()
    
    def evaluate(self, reference, hypothesis): # Change this to file names
        '''
        Computes the Diarization Error Rate.
        
        Args:
            reference: the ground truth
            hypothesis: the predicted labels
        Returns:
            int: diarization error rate
        '''
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
        
    def visualize(self, output_path):
        """
        Creates and saves a heatmap of the results.
        
        Args:
            output_path: location where the graph will be saved
        """
        pass    
    