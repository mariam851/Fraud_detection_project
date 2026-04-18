# src/losses/focal_loss.py
import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    """
    Research-Grade Focal Loss for Imbalanced Financial Data.
    Designed to force the model to focus on 'Hard-to-Classify' Fraudulent samples.
    
    Why this matters in Research:
    - Standard Cross-Entropy gets overwhelmed by the majority (Legitimate transactions).
    - Focal Loss reshapes the loss curve to penalize easy-to-classify examples.
    """
    def __init__(self, alpha=0.999, gamma=2):
        """
        Args:
            alpha (float): Balancing factor for the rare class (Fraud). 
                        We use 0.999 because fraud is extremely rare (~0.1%).
            gamma (float): Focusing parameter. 
                        Reduces loss for well-classified examples, focusing on errors.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        # Use reduction='none' to apply the weight manually per sample
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        # 1. Calculate Standard Binary Cross Entropy
        bce_loss = self.bce(inputs, targets)
        
        # 2. Calculate Probability (pt) from the logit
        # pt represents how confident the model is in its prediction
        pt = torch.exp(-bce_loss) 
        
        # 3. Calculate Focal Loss Formula:
        # FL(pt) = -alpha * (1 - pt)^gamma * log(pt)
        # (1 - pt)^gamma is the 'modulating factor' that shrinks loss for easy samples
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        # Return the mean of the loss across the batch
        return focal_loss.mean()

    def __repr__(self):
        """String representation for research documentation."""
        return f"FocalLoss(alpha={self.alpha}, gamma={self.gamma})"
