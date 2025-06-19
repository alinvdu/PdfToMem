from dataclasses import dataclass

@dataclass
class MCPConfig:
    """High‑level knobs exposed to pipeline consumers."""

    extract_and_embed_images: bool = True
    look_for_queryable_tables: bool = True
