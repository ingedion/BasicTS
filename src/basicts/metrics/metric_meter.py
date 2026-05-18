class AvgMeter:
    """Average meter.
    """

    def __init__(self):
        self._sum: float = 0.
        self._count: int = 0
        self.last: float = 0.

    def reset(self):
        """Reset counter.
        """

        self._sum = 0.
        self._count = 0
        self.last = 0.

    def update(self, value: float, n: int = 1):
        """Update sum and count.

        Args:
            value (float): value.
            n (int): number.
        """

        self._sum += value * n
        self._count += n
        self.last = value

    @property
    def value(self) -> float:
        """Get average value.

        Returns:
            avg (float)
        """

        return self._sum / self._count if self._count != 0 else 0


class RMSEMeter:
    """
    RMSE meter.
    This meter maintains **MSE** and calculate **RMSE** in the post process.
    """

    def __init__(self):
        self._mse: float = 0.
        self._count: int = 0

    def reset(self):
        """Reset counter.
        """

        self._mse = 0.
        self._count = 0

    def update(self, value: float, n: int = 1):
        """Update sum and count.

        Args:
            value (float): value.
            n (int): number.
        """

        self._mse += value ** 2 * n
        self._count += n

    @property
    def value(self) -> float:
        """Get average value.

        Returns:
            avg (float)
        """

        mse = self._mse / self._count if self._count != 0 else 0

        return mse ** 0.5


class R2Meter:
    """
    R² meter with robust accumulation.
    
    R² is not a simple additive metric — it depends on global variance (ss_tot).
    When computed per-batch and averaged, batches with near-zero local variance
    produce extreme R² values (e.g., -1e7) that dominate the average.
    
    This meter clips per-batch R² to [-100, 1] before accumulation, preventing
    degenerate batches from corrupting the epoch-level statistic. The clipping
    bound of -100 is generous enough to preserve meaningful negative R² values
    (model worse than mean baseline) while filtering out numerical artifacts.
    
    For exact epoch-level R², one would need to accumulate raw ss_res and ss_tot
    across batches, which requires changes to the runner interface. This clipped
    approach is a practical compromise within the existing (value, n) interface.
    """

    def __init__(self):
        self._sum: float = 0.
        self._count: int = 0
        self.last: float = 0.

    def reset(self):
        self._sum = 0.
        self._count = 0
        self.last = 0.

    def update(self, value: float, n: int = 1):
        """Update with per-batch R² value, clipped to [-100, 1].

        Args:
            value (float): Per-batch R² value.
            n (int): Number of valid elements in this batch.
        """

        self.last = value
        clipped = max(min(value, 1.0), -100.0)
        self._sum += clipped * n
        self._count += n

    @property
    def value(self) -> float:
        """Get weighted average R² (with clipping applied during accumulation)."""

        return self._sum / self._count if self._count != 0 else 0.0
