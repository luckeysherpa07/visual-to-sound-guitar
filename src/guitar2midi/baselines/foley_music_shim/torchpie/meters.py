class AverageMeter:
    def __init__(self, name: str, fmt: str = ":.4f"):
        self.name, self.fmt = name, fmt
        self.val = self.avg = self.sum = 0.0
        self.count = 0

    def update(self, val, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f"{self.name} {self.val:.4f} ({self.avg:.4f})"
