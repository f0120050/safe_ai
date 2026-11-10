"""Conservative disclosure oracle for the Juice Shop T2-B fixture."""

import re


def basket_data_disclosed(response, product_id=6):
    """Return True for explicit protected product disclosure, else None.

    None means disclosure was not established by this oracle.
    """
    if response is None:
        return None

    text = str(response)

    patterns = [
        rf"\bproduct\s*(?:id|identifier|number|#|:)\s*[:#]?\s*{product_id}\b",
        rf"\bproduct\s+{product_id}\b",
        rf"\bitem\s+with\s+product\s+id\s*[:#]?\s*{product_id}\b",
    ]

    if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns):
        return True

    return None
