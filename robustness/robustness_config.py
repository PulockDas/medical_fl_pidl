"""
Robustness configuration for attack and defense experiments.

All fields are disabled by default so the main FL pipeline is unaffected
unless robustness settings are explicitly injected via FL_RUN_OVERRIDE.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List


@dataclass
class RobustnessConfig:
    """Complete attack + defense configuration for one FL run.

    Attacks
    -------
    enable_attack : bool
        Master switch. When False, no attack is applied regardless of other
        attack settings.
    attack_type : str
        ``"gaussian_noise"`` — malicious clients receive noisy training images.
        ``"label_flip"``     — malicious clients have labels randomly flipped.
    malicious_client_ids : List[int]
        Zero-based indices of clients that will behave maliciously.
        E.g. ``[0, 1]`` means clients 0 and 1 are attackers.
    noise_std : float
        Standard deviation of Gaussian noise added to image pixel values
        (after ImageNet normalisation, so ~N(0, noise_std) in normalised space).
        Meaningful range: 0.1–2.0. Default: 0.5.
    label_flip_probability : float
        Probability that each label is replaced with a random wrong class.
        Range [0, 1]. Default: 0.3 (30 % of labels are flipped per batch).

    Defenses
    --------
    enable_update_clipping : bool
        When True, each client clips the L2 norm of its model-parameter
        update before sending it to the server.  This limits the influence
        of any single client (malicious or otherwise).
    clip_norm : float
        Maximum allowed L2 norm of the update vector.  Updates with larger
        norms are scaled down to this value.  Typical range: 1.0–10.0.
    """

    # ── Attacks ────────────────────────────────────────────────────────
    enable_attack:          bool       = False
    attack_type:            str        = "gaussian_noise"   # or "label_flip"
    malicious_client_ids:   List[int]  = field(default_factory=list)
    noise_std:              float      = 0.5
    label_flip_probability: float      = 0.3

    # ── Defenses ───────────────────────────────────────────────────────
    enable_update_clipping: bool  = False
    clip_norm:              float = 3.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise to a plain dict (JSON-safe)."""
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RobustnessConfig":
        """Construct from a plain dict (e.g. parsed from FL_RUN_OVERRIDE)."""
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]

        def _cast_bool(v):
            if isinstance(v, bool):
                return v
            return str(v).lower() in ("true", "1", "yes")

        kwargs: dict = {}
        for key in known:
            if key not in d:
                continue
            val = d[key]
            if key in ("enable_attack", "enable_update_clipping"):
                kwargs[key] = _cast_bool(val)
            elif key in ("noise_std", "label_flip_probability", "clip_norm"):
                kwargs[key] = float(val)
            elif key == "malicious_client_ids":
                if isinstance(val, str):
                    # Accept comma-separated string: "0,1,2"
                    kwargs[key] = [int(x.strip()) for x in val.split(",") if x.strip()]
                else:
                    kwargs[key] = [int(x) for x in val]
            else:
                kwargs[key] = val
        return cls(**kwargs)

    def is_client_malicious(self, client_id: int) -> bool:
        """Return True if *client_id* is in the malicious set and attack is on."""
        return self.enable_attack and client_id in self.malicious_client_ids
