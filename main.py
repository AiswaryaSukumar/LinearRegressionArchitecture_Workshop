import os

# Import pipeline components
from DataExtractionAnalysis.DataExtractor import DataExtractor
from DataExtractionAnalysis.DataPreparation import DataPreparator
from ModelTraining.ModelTrainer import ModelTrainer
from ModelTraining.ModelSelector import ModelSelector
from ModelEvaluation.Threshold import ThresholdCalculator
from ModelEvaluation.ThresholdChecker import EventDetector
from PredictionService.StreamingSimulator import StreamingSimulator
from Visualization.plot import Plotter


class Orchestrator:
    def __init__(self):
        # Shared pipeline state
        self.data = None
        self.models = None
        self.residuals_dict = None
        self.thresholds = None
        self.events = None
        self.residuals_test = None

    def run(self):
        print(" Starting Predictive Maintenance Pipeline...\n")

        # Step 1: Extract Data
        extractor = DataExtractor()
        self.data = extractor.extract()
        print(" Data extraction complete.")

        # Step 2: Prepare Data
        preparator = DataPreparator()
        self.data = preparator.prepare(self.data)
        print("Data preparation complete.")

        # Step 3: Model Selection + Training
        selector = ModelSelector()
        chosen_model = selector.select_model()   # Stub
        trainer = ModelTrainer()
        self.models, self.residuals_dict = trainer.train(self.data)
        print(f" Model training complete. Using: {chosen_model}")

        # Step 4: Threshold Calculation
        threshold_calc = ThresholdCalculator()
        self.thresholds = threshold_calc.calculate(self.residuals_dict)
        print(" Thresholds discovered.")

        # Step 5: Streaming Simulation
        simulator = StreamingSimulator(self.data, self.models)
        self.residuals_test = simulator.run_streaming()
        print(" Streaming simulation complete.")

        # Step 6: Event Detection
        detector = EventDetector()
        self.events = detector.detect(self.data, self.models, self.thresholds)
        print(" Events detected (alerts & errors).")

        # Step 7: Visualization
        plotter = Plotter()
        plotter.plot_all(self.data, self.residuals_test, self.thresholds, self.events)
        print(" Visualization complete.")

        print("\n Pipeline finished. Results saved in /data and /images.")


if __name__ == "__main__":
    orchestrator = Orchestrator()
    orchestrator.run()
