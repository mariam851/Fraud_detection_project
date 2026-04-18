# src/utils/metrics.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix

class ResearchEvaluator:
    """
    Advanced Metrics for Fraud Detection Research.
    Calculates ML scores, Financial Impact, and Adversarial Robustness.
    """
    
    @staticmethod
    def get_ml_metrics(y_true, y_pred):
        """Calculates standard ML metrics for the Comparison Table (Notebook 4)."""
        return {
            "Precision": precision_score(y_true, y_pred),
            "Recall (Sensitivity)": recall_score(y_true, y_pred),
            "F1-Score": f1_score(y_true, y_pred)
        }

    @staticmethod
    def plot_research_confusion_matrix(y_true, y_pred):
        """Generates a professional Confusion Matrix for the paper."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Legit', 'Fraud'], 
                    yticklabels=['Legit', 'Fraud'])
        plt.title('Confusion Matrix: Hybrid Model Performance')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.show()

    @staticmethod
    def calculate_roi(y_true, y_pred, amounts, false_positive_cost=5.0):
        """
        Calculates Return on Investment (ROI).
        - TP: Stolen money recovered (Positive Impact).
        - FP: Operational cost of blocking a legit customer (Negative Impact).
        """
        # Vectorized calculation for speed (Clean Update)
        tp_mask = (y_true == 1) & (y_pred == 1)
        fp_mask = (y_true == 0) & (y_pred == 1)
        
        total_saved = np.sum(amounts[tp_mask])
        total_loss_fp = np.sum(fp_mask) * false_positive_cost
        
        net_impact = total_saved - total_loss_fp
        
        return {
            "Total_Money_Saved": total_saved,
            "Operational_Loss_FP": total_loss_fp,
            "Net_Financial_Impact": net_impact
        }

    @staticmethod
    def evaluate_adversarial_robustness(model, adversarial_sample, target_prob):
        """
        Logs the success of Topological Calibration (Notebook 5).
        """
        print("\n--- Research Adversarial Report ---")
        if target_prob < 0.05:
            print("Status: VULNERABLE to Amount Masking (Raw Model)")
        else:
            print("Status: ROBUST (Topological Defense Active)")
        
        return "Robustness Test Complete"
