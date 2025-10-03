from dataclasses import dataclass


@dataclass
class HRMCoreConfig:
    vocab_size: int
    hidden_size: int = 768
    num_heads: int = 12
    ff_mult: int = 4
    max_position_embeddings: int = 512

    # Hierarchical controller
    H_layers: int = 2
    L_layers: int = 2
    H_cycles: int = 2
    L_cycles: int = 1

    # I/O
    tie_embeddings: bool = False