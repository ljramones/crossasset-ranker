"""Active-track data package.

Only ``market_cache`` lives here. ``data.market_data`` is intentionally absent —
the legacy model-zoo CLI (``main.py``, ``utils/experiment.py``,
``audit/integrity_audit.py``) imports it, but those paths are frozen and must
not be revived as part of the post-reset active research track.
"""
