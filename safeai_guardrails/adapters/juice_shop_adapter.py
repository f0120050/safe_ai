"""HTTP adapter for the local OWASP Juice Shop laboratory."""


class JuiceShopAdapter:
    def __init__(self, base_url, session, authorized_basket_id):
        self.base_url = base_url.rstrip("/")
        self.session = session
        self.authorized_basket_id = int(authorized_basket_id)

    def get_my_basket(self):
        """Retrieve the authenticated user's basket."""
        response = self.session.get(
            f"{self.base_url}/rest/basket/{self.authorized_basket_id}",
            timeout=15,
        )
        response.raise_for_status()

        basket = response.json()["data"]

        if int(basket["id"]) != self.authorized_basket_id:
            raise ValueError("Unexpected authorized basket ID")

        return basket

    def get_basket_item(self, item_id):
        """Retrieve a basket item without making an authorization decision."""
        response = self.session.get(
            f"{self.base_url}/api/BasketItems/{int(item_id)}",
            timeout=15,
        )
        response.raise_for_status()

        item = response.json()["data"]

        if int(item["id"]) != int(item_id):
            raise ValueError("Unexpected basket item ID")

        return item

    def is_my_basket_item(self, item_id):
        """Check whether an item belongs to the authorized basket."""
        item = self.get_basket_item(item_id)
        return int(item["BasketId"]) == self.authorized_basket_id
