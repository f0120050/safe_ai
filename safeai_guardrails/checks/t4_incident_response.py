"""T4 incident response: safe rollback and simulated notifications."""

from dataclasses import dataclass


@dataclass
class IncidentResponseResult:
    rollback_success: bool
    notification_created: bool
    reason: str


class T4IncidentResponse:
    def __init__(self, authorized_target_session, notification_sink):
        self.session = authorized_target_session
        self.notification_sink = notification_sink

    def respond(
        self,
        base_url,
        item_id,
        target_basket_id,
        previous_quantity,
        unauthorized_quantity,
    ):
        url = f"{base_url.rstrip('/')}/api/BasketItems/{int(item_id)}"

        response = self.session.get(url, timeout=15)
        response.raise_for_status()
        item = response.json()["data"]

        if int(item["BasketId"]) != int(target_basket_id):
            return IncidentResponseResult(
                False, False, "Target ownership mismatch"
            )

        if int(item["quantity"]) != int(unauthorized_quantity):
            self.notification_sink.append({
                "event": "UNAUTHORIZED_BASKET_MODIFICATION",
                "target_basket_id": int(target_basket_id),
                "item_id": int(item_id),
                "recipients": ["shop_administrator", "affected_customer"],
                "rollback_success": False,
                "manual_review_required": True,
                "simulated": True,
            })
            return IncidentResponseResult(
                False, True,
                "State changed; automatic rollback unsafe; notification created"
            )

        restored = False

        try:
            update = self.session.put(
                url,
                json={"quantity": int(previous_quantity)},
                timeout=15,
            )
            update.raise_for_status()

            verify = self.session.get(url, timeout=15)
            verify.raise_for_status()
            restored_item = verify.json()["data"]

            restored = (
                int(restored_item["BasketId"]) == int(target_basket_id)
                and int(restored_item["quantity"])
                == int(previous_quantity)
            )
        except Exception:
            restored = False

        self.notification_sink.append({
            "event": "UNAUTHORIZED_BASKET_MODIFICATION",
            "target_basket_id": int(target_basket_id),
            "item_id": int(item_id),
            "recipients": ["shop_administrator", "affected_customer"],
            "rollback_success": restored,
            "simulated": True,
        })

        return IncidentResponseResult(
            rollback_success=restored,
            notification_created=True,
            reason=(
                "Rollback verified"
                if restored else "Rollback failed; manual review required"
            ),
        )
