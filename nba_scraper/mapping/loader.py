from typing import Dict, Tuple, Optional

import yaml

from .descriptor_norm import canon_str

SignatureKey = Tuple[str, str, str, Tuple[str, ...]]


def load_mapping(path: Optional[str]) -> Dict[SignatureKey, Dict[str, object]]:
    """Load a curated mapping YAML (derived from mapping_template.yml).

    Parameters
    ----------
    path:
        Optional path to a YAML file describing overrides.

    Returns
    -------
    dict
        Mapping keyed by ``(family, subType_norm, descriptor_core, qualifiers_tuple)``
        with override dictionaries such as ``{"eventmsgactiontype": int}`` or
        ``{"subfamily": "str"}``.
    """
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    table: Dict[SignatureKey, Dict[str, object]] = {}
    for group in (data or []):
        fam = canon_str(group.get("actionType"))
        for signature in group.get("signatures", []):
            sig = signature.get("signature", {})
            st = canon_str(sig.get("subType"))
            desc = canon_str(sig.get("descriptor"))
            quals = tuple(sorted(canon_str(q) for q in sig.get("qualifiers", [])))
            key = (fam, st, desc, quals)
            overrides: Dict[str, object] = {}
            if signature.get("map_to_event_name"):
                overrides["subfamily"] = str(signature["map_to_event_name"]).lower()
            if signature.get("map_to_msg_action"):
                overrides["eventmsgactiontype"] = int(signature["map_to_msg_action"])
            if signature.get("map_to_msg_type"):
                overrides["eventmsgtype"] = int(signature["map_to_msg_type"])
            table[key] = overrides
    return table
