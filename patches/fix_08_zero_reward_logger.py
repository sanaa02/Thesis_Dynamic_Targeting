"""
FIX 08 — env_alsat_dynamic.py  diagnostic logging for zero-reward static actions
==================================================================================
When a static imaging action returns zero reward, log the specific reason:
  - CLOUDY: cloud_truth >= CLOUD_THRESH
  - NO_IMAGE: was_image_taken returned False (bsk_rl didn't confirm imaging)
  - SLEW_LIMIT: off-nadir angle exceeded 45°

This is injected into DynamicScienceDataStore.compare_log_states() just
after the static target reward calculation, so it fires only for static
actions that produce zero reward.  No change to reward values.
"""

OLD = '''        logger.debug(
            f"[STATIC] image taken: target={target.name}  "
            f"cloud_truth={cloud_truth:.2f}  priority={priority:.2f}  "
            f"slew_deg={math.degrees(slew_angle):.1f}  reward={reward:+.4f}"
        )'''

NEW = '''        logger.debug(
            f"[STATIC] image taken: target={target.name}  "
            f"cloud_truth={cloud_truth:.2f}  priority={priority:.2f}  "
            f"slew_deg={math.degrees(slew_angle):.1f}  reward={reward:+.4f}"
        )
        # ── FIX-08: Zero-reward static diagnostic ─────────────────────────
        if reward <= 0.0 and not is_dynamic:
            _reason_08 = (
                "CLOUDY" if cloud_truth >= CLOUD_THRESH
                else "SLEW_LIMIT" if math.degrees(slew_angle) > 45.0
                else "NEGATIVE_SLEW_COST"
            )
            logger.debug(
                f"[STATIC-ZERO] target={target.name}  reason={_reason_08}  "
                f"cloud={cloud_truth:.2f}  slew_deg={math.degrees(slew_angle):.1f}  "
                f"reward={reward:+.4f}"
            )
        # ──────────────────────────────────────────────────────────────────'''


def apply(text: str) -> str:
    if OLD not in text:
        raise ValueError(
            "FIX-08: STATIC image-taken logger not found in env_alsat_dynamic.py."
        )
    return text.replace(OLD, NEW, 1)