import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


class Evaluator:
    def __init__(self, model_name: str = None):
        self.model_name: str = model_name
        pass
        
    def print_evaluation(self, predictions, targets, labels: list[str], show_plot=False):
        print(classification_report(targets, predictions))

        # Confusion Matrix
        conf_matrix = confusion_matrix(targets, predictions)
        display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=labels)
        display.plot()
        if self.model_name:
            plt.title(self.model_name)
        if show_plot:
            plt.show()
        print('Confusion Matrix:\n', conf_matrix)
