"""Juice Shop adapter, version 2.

Ownership is derived only from the authenticated user's own basket,
read through the authorized basket endpoint. Unlike v1, the target
object is never read, so the check does not itself exercise the BOLA
vulnerability it is meant to guard against.
"""

from safeai_guardrails.adapters.juice_shop_adapter import JuiceShopAdapter


class JuiceShopAdapterV2(JuiceShopAdapter):
    def my_basket_item_ids(self):
        basket = self.get_my_basket()  # validates the authorized basket ID
        item_ids = set()

        for product in basket.get("Products") or []:
            item = product.get("BasketItem") or {}
            if "id" not in item or "BasketId" not in item:
                raise ValueError("Unexpected basket item structure")
            if int(item["BasketId"]) != self.authorized_basket_id:
                raise ValueError("Basket item outside the authorized basket")
            item_ids.add(int(item["id"]))

        return item_ids

    def is_my_basket_item(self, item_id):
        return int(item_id) in self.my_basket_item_ids()
