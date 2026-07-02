# SPDX-FileCopyrightText: 2026 Judder <judder@forven.app>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Canonical agent roster helpers.

Single source of truth for agent-id normalization (retired/legacy agent ids
mapped to their living successors). The full roster definition (seeds, roles,
channels, ownership) is being consolidated here — import from this module
instead of duplicating alias maps.
"""

# Retired/legacy agent ids → the agent that inherited their duties.
LEGACY_AGENT_ALIASES = {
    "backtest-engineer": "simulation-agent",
    "system": "brain",
}


def normalize_agent_id(agent_id: str | None) -> str:
    """Lowercase and map retired/legacy agent ids to their successors."""
    normalized = str(agent_id or "").strip().lower()
    return LEGACY_AGENT_ALIASES.get(normalized, normalized)
