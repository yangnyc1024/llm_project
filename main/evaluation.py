

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

class ModelEvaluator:
    def __init__(self):
        self.labels = ['positive', 'neutral', 'negative']
        self.mapping = {'positive': 2, 'neutral': 1, 'none': 1, 'negative': 0}

    def map_labels(self, labels):
        """Maps textual labels to numeric values based on predefined mapping."""
        map_func = np.vectorize(lambda x: self.mapping.get(x, 1))
        return map_func(labels)

    def calculate_accuracy(self, y_true, y_pred):
        """Calculates and prints the overall accuracy."""
        accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
        print(f'Accuracy: {accuracy:.3f}')

    def accuracy_per_label(self, y_true, y_pred):
        """Calculates and prints the accuracy for each unique label."""
        unique_labels = set(y_true)
        for label in unique_labels:
            label_indices = [i for i in range(len(y_true)) if y_true[i] == label]
            label_y_true = [y_true[i] for i in label_indices]
            label_y_pred = [y_pred[i] for i in label_indices]
            accuracy = accuracy_score(label_y_true, label_y_pred)
            print(f'Accuracy for label {label}: {accuracy:.3f}')

    def generate_reports(self, y_true, y_pred):
        """Generates and prints classification report and confusion matrix."""
        print('\nClassification Report:')
        print(classification_report(y_true=y_true, y_pred=y_pred))
        print('\nConfusion Matrix:')
        conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1, 2])
        print(conf_matrix)

    def evaluate(self, y_true: pd.Series, y_pred: list):
        """Main evaluation method that orchestrates the evaluation process."""
        y_true_mapped = self.map_labels(y_true)
        y_pred_mapped = self.map_labels(y_pred)

        self.calculate_accuracy(y_true_mapped, y_pred_mapped)
        self.accuracy_per_label(y_true_mapped, y_pred_mapped)
        self.generate_reports(y_true_mapped, y_pred_mapped)

# # Usage
# evaluator = ModelEvaluator()
# evaluator.evaluate(y_true, y_pred)
