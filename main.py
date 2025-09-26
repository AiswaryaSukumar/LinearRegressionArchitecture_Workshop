import os
from DataExtractionAnalysis.DataExtraction import DataExtractor
from DataPreparation.DataPreparation import Data_preparation
from ModelTraining.ModelTraining import ModelTrainer
from ModelEvaluation.Threshold import ThresholdFinder
from ModelEvaluation.ThresholdChecker import EventDetector
from PredictionService.StreamingSimulator import StreamingSimulator
from Visualization.plot import Plotter

class Orchestrator:
    def __init__(self):
        self.data = None
        self.models = None
        self.residuals_dict = None
        self.thresholds = None
        self.events = None
        self.residuals_test = None

    def run(self):
        print("🚀 Starting Predictive Maintenance Pipeline...\n")

        # Step 1: Extract Data
        extractor = DataExtractor()
        self.data = extractor.extract()

        # Step 2: Prepare Data
        preparator = DataPreparator()
        self.data = preparator.prepare(self.data)

        # Step 3: Train Models
        trainer = ModelTrainer()
        self.models, self.residuals_dict = trainer.train(self.data)

        # Step 4: Discover Thresholds
        discover_thresholds = ThresholdFinder()
        self.thresholds = discover_thresholds.calculate(self.residuals_dict)

        # Step 5: Simulate Streaming & Collect Residuals
        simulator = StreamingSimulator(self.data, self.models)
        self.residuals_test = simulator.run_streaming()

        # Step 6: Detect Events (Alerts/Errors)
        detector = EventDetector()
        self.events = detector.detect(self.data, self.models, self.thresholds)

        # Step 7: Visualize
        plotter = Plotter()
        plotter.plot_all(self.data, self.residuals_test, self.thresholds, self.events)

        print("\n✅ Pipeline finished. Results saved in /data and /images.")


if __name__ == "__main__":
    orchestrator = Orchestrator()
    orchestrator.run()
