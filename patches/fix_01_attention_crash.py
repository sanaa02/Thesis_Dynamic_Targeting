"""
FIX 01 — train_full_system.py  --attention crash
=================================================
Root cause: `features_extractor_kwargs` used `embed_dim=64` but
`SchedulerAttentionExtractor.__init__` expects `d_model`.  SB3 passes
the kwargs directly to the constructor, so the unknown kwarg crashes.

Second issue: `net_arch = dict(pi=..., vf=...)` is only valid for SB3
ActorCriticPolicy when you want separate network widths; using it with a
custom extractor causes an unexpected-keyword TypeError.  Since the
attention extractor already compresses obs to 256-d, `net_arch=[]` is
the correct setting.
"""

OLD = '''            policy_kwargs = dict(
                features_extractor_class  = SchedulerAttentionExtractor,
                features_extractor_kwargs = dict(n_heads=4, embed_dim=64),
                net_arch = dict(pi=[256, 128], vf=[256, 128]),
            )'''

NEW = '''            policy_kwargs = dict(
                features_extractor_class  = SchedulerAttentionExtractor,
                features_extractor_kwargs = dict(features_dim=256, d_model=64, n_heads=4),
                net_arch = [],   # extractor handles all feature compression → no extra MLP needed
            )'''


def apply(text: str) -> str:
    if OLD not in text:
        raise ValueError(
            "FIX-01: expected pattern not found in train_full_system.py.\n"
            "The file may have already been patched or changed.\n"
            f"Looking for:\n{OLD}"
        )
    return text.replace(OLD, NEW, 1)