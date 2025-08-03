import copy, random, torch
from collections import deque

class CheckpointPool:
    """
    Keeps the K best checkpoints, ranked by a scalar skill score
    (higher = better).  Sampling is soft-max-weighted by score.
    """
    def __init__(self, max_size: int = 5):
        self.max_size = max_size
        self.pool: deque[tuple[float, torch.nn.Module]] = deque()  # (score, model)

    # ------------------------------------------------------------ #

    def add(self, model, score):
        """Insert a deepcopy of `model` if it’s good enough."""
        self.pool.append((score, copy.deepcopy(model)))
        # keep best first
        self.pool = deque(sorted(self.pool, key=lambda x: x[0], reverse=True))

        # trim worst
        while len(self.pool) > self.max_size:
            self.pool.pop()

    # ------------------------------------------------------------ #

    def sample(self) -> torch.nn.Module:
        """Return a random checkpoint, weighted by skill."""
        if not self.pool:
            raise RuntimeError("Checkpoint pool empty!")

        scores, models = zip(*self.pool)
        probs = torch.softmax(torch.tensor(scores, dtype=torch.float32), dim=0).tolist()
        return random.choices(models, weights=probs, k=1)[0]

    def report(self):

        if not self.pool:
            msg = "CheckpointPool is empty."
            print(msg)
            return

        rows = []
        for rank, (score, model) in enumerate(self.pool, start=1):
            n_params = sum(p.numel() for p in model.parameters()) / 1e6  # in millions
            rows.append(f"│{rank:^7}│{score:^20.2f}")

        header = ("Rank - Skill\n"
                  + "\n".join(rows) +
                  "\n")

        print(header)

    def __len__(self):
        return len(self.pool)

    def best_score(self):
        return self.pool[0][0] if self.pool else None
