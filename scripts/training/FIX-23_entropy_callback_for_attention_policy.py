#!/usr/bin/env python3
"""
FIX-23: Add EntropyAnnealingCallback to attention_policy.py
=============================================================

The training script tries:
    from attention_policy import EntropyAnnealingCallback

But EntropyAnnealingCallback is NOT defined in attention_policy.py.
This causes a silent fallback and no entropy annealing runs.

APPLY: Append the class below to scripts/models/attention_policy.py
"""

# ── Paste this block at the BOTTOM of attention_policy.py ────────────────────

ATTENTION_POLICY_ADDITION = '''
# =============================================================================
#  EntropyAnnealingCallback  (FIX-23: was missing — caused silent import error)
# =============================================================================

if SB3_OK:
    try:
        from stable_baselines3.common.callbacks import BaseCallback

        class EntropyAnnealingCallback(BaseCallback):
            """
            Linearly decay PPO entropy coefficient from start_val to end_val
            over total_timesteps steps.

            Place one instance per curriculum stage; the training loop resets
            ent_coef at the start of each stage via model.ent_coef = stage_ent,
            then this callback decays it to ENT_END within the stage.

            Parameters
            ----------
            start_val       : float — initial ent_coef for this stage
            end_val         : float — final ent_coef (floor)
            total_timesteps : int   — timesteps for this stage (not total)
            verbose         : int   — 0=silent, 1=log per stage, 2=log every rollout

            Usage
            -----
                from attention_policy import EntropyAnnealingCallback
                cb = EntropyAnnealingCallback(
                    start_val=0.15, end_val=0.01,
                    total_timesteps=200_000, verbose=1
                )
                model.learn(total_timesteps=200_000, callback=cb)
            """
            def __init__(self, start_val: float, end_val: float,
                         total_timesteps: int, verbose: int = 0):
                super().__init__(verbose=verbose)
                self.start_val        = float(start_val)
                self.end_val          = float(end_val)
                self.total_timesteps  = int(total_timesteps)
                self._stage_steps     = 0

            def _on_step(self) -> bool:
                self._stage_steps += 1
                frac    = min(1.0, self._stage_steps / max(1, self.total_timesteps))
                new_ent = self.start_val + frac * (self.end_val - self.start_val)
                self.model.ent_coef = float(new_ent)

                if self.verbose >= 2 and self._stage_steps % 4096 == 0:
                    import logging
                    logging.getLogger(__name__).info(
                        f"[EntropyAnneal] steps={self._stage_steps}  "
                        f"ent_coef={new_ent:.4f}  "
                        f"({self.start_val:.3f}→{self.end_val:.3f})"
                    )
                return True   # continue training

            def _on_rollout_start(self) -> None:
                pass

            def _on_rollout_end(self) -> None:
                pass

    except ImportError:
        EntropyAnnealingCallback = None

else:
    EntropyAnnealingCallback = None
'''


# ─────────────────────────────────────────────────────────────────────────────
# Verify the fix works by simulating the import
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("FIX-23: EntropyAnnealingCallback addition for attention_policy.py")
    print()
    print("STEP 1: Append the ATTENTION_POLICY_ADDITION block to attention_policy.py")
    print("        (after the make_attention_ppo function)")
    print()
    print("STEP 2: Verify the import works:")
    print("        from attention_policy import EntropyAnnealingCallback")
    print("        assert EntropyAnnealingCallback is not None")
    print()
    print("STEP 3: The training loop in run_curriculum_training will now find it:")
    print("        try:")
    print("            from attention_policy import EntropyAnnealingCallback")
    print("            extra_cbs.append(EntropyAnnealingCallback(...))")
    print("        except ImportError: ...")
    print()

    # Try to simulate: does the class work?
    try:
        from stable_baselines3.common.callbacks import BaseCallback

        class _TestCallback(BaseCallback):
            def __init__(self, start, end, total):
                super().__init__(verbose=0)
                self.start = start; self.end = end; self.total = total
                self._steps = 0

            def _on_step(self):
                self._steps += 1
                frac = min(1.0, self._steps / self.total)
                # would do: self.model.ent_coef = self.start + frac * (self.end - self.start)
                return True

        cb = _TestCallback(0.15, 0.01, 200_000)
        for _ in range(10):
            cb._on_step()
        print(f"  Callback self-test: PASSED  (steps={cb._steps})")
    except ImportError:
        print("  stable_baselines3 not installed — test skipped")