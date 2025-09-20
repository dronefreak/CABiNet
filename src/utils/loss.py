#!/usr/bin/python
# -*- encoding: utf-8 -*-

# src/utils/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class OhemCELoss(nn.Module):
    """Online Hard Example Mining Cross Entropy Loss with optional class weights.

    Minimizes top-k hard examples above threshold, or n_min hardest if below.
    Handles edge cases: empty valid masks, device placement, mixed precision.
    """

    def __init__(self, thresh, n_min, ignore_lb=255, weight=None):
        """
        Args:
            thresh: float, threshold for loss (before -log(thresh))
            n_min: int, minimum number of pixels to select
            ignore_lb: int, label to ignore
            weight: Tensor of shape (C,), optional weight for each class
        """
        super(OhemCELoss, self).__init__()
        self.thresh = float(thresh)
        self.n_min = int(n_min)
        self.ignore_lb = ignore_lb
        self.register_buffer("weight", weight)  # Safe device transfer

        # Use reduction='none' to get per-pixel loss
        self.criteria = nn.CrossEntropyLoss(
            ignore_index=ignore_lb,
            reduction="none",
            weight=weight,  # Apply class weights here
        )

    def forward(self, logits, labels):
        """
        Args:
            logits: (N, C, H, W)
            labels: (N, H, W)
        Returns:
            scalar loss
        """
        # Compute per-pixel CE loss (already ignores `ignore_lb`)
        loss = self.criteria(logits, labels)  # (N, H, W)

        # Flatten and remove ignored entries
        valid_mask = labels != self.ignore_lb  # (N, H, W)
        valid_loss = loss[valid_mask]

        # Handle case where no valid pixels exist
        if valid_loss.numel() == 0:
            return torch.zeros((), device=logits.device, requires_grad=True)

        # Sort descending
        valid_loss_sorted, _ = torch.sort(valid_loss, descending=True)

        # Clamp n_min to available number of valid pixels
        n_min = min(self.n_min, valid_loss_sorted.numel())

        # Choose top-k: either those above threshold, or top n_min
        if valid_loss_sorted[n_min - 1] > self.thresh:
            selected_loss = valid_loss_sorted[valid_loss_sorted > self.thresh]
        else:
            selected_loss = valid_loss_sorted[:n_min]

        # Return mean of selected hard examples
        return selected_loss.mean()

    def extra_repr(self):
        return f"thresh={self.thresh}, n_min={self.n_min}, ignore_lb={self.ignore_lb}"


class SoftmaxFocalLoss(nn.Module):
    """Focal Loss based on softmax probability.

    More numerically stable than applying exp manually.
    """

    def __init__(self, gamma, ignore_lb=255):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_lb = ignore_lb
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        # Apply softmax over classes
        prob = F.softmax(logits, dim=1)  # (N, C, H, W)
        # Compute log_softmax for numerical stability
        log_prob = F.log_softmax(logits, dim=1)
        # Modulate log-prob with (1-p)^gamma
        weight = (1 - prob) ** self.gamma
        # Weighted log likelihood
        focal_log_prob = weight * log_prob
        # NLL over weighted distribution
        loss = self.nll(focal_log_prob, labels)
        return loss


if __name__ == "__main__":
    torch.manual_seed(15)

    # Config
    n_classes = 19
    batch_size = 16
    H = W = 20
    n_min = batch_size * H * W // 16  # 400

    # Create OHEM loss
    criteria1 = OhemCELoss(thresh=0.7, n_min=n_min).cuda()
    criteria2 = OhemCELoss(thresh=0.7, n_min=n_min).cuda()

    # Dummy network
    net1 = (
        nn.Sequential(nn.Conv2d(3, n_classes, kernel_size=3, stride=1, padding=1))
        .cuda()
        .train()
    )
    net2 = (
        nn.Sequential(nn.Conv2d(3, n_classes, kernel_size=3, stride=1, padding=1))
        .cuda()
        .train()
    )

    # Input and labels
    inten = torch.randn(batch_size, 3, H, W).cuda()
    lbs = torch.randint(0, n_classes, (batch_size, H, W)).cuda()
    # Simulate some ignored regions
    lbs[0] = 255  # First sample fully ignored
    lbs[1, :5, :5] = 255  # Partial ignore

    # Forward pass
    logits1 = net1(inten)
    logits2 = net2(inten)

    # Interpolate if needed (optional)
    logits1 = F.interpolate(logits1, size=(H, W), mode="bilinear", align_corners=False)
    logits2 = F.interpolate(logits2, size=(H, W), mode="bilinear", align_corners=False)

    # Compute losses
    loss1 = criteria1(logits1, lbs)
    loss2 = criteria2(logits2, lbs)
    loss = loss1 + loss2

    print("Total Loss:", loss.detach().cpu().item())

    # Backward
    loss.backward()
    print("✅ Backward passed successfully.")
