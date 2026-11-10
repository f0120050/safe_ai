
import getpass
import requests

BASE_URL = "http://127.0.0.1:8888"


def main():
    email = input("User Y email: ").strip()
    password = getpass.getpass("Password: ")

    session = requests.Session()
    session.trust_env = False

    try:
        login_response = session.post(
            f"{BASE_URL}/identity/api/auth/login",
            json={
                "email": email,
                "password": password
            },
            timeout=15
        )

        print("Login status:", login_response.status_code)

        login_data = login_response.json()

        if login_response.status_code != 200:
            print("Login failed:", login_data.get("message"))
            return

        if login_data.get("isMfaRequired"):
            print("MFA is required. Stopping.")
            return

        token = login_data.get("token")

        if not token:
            print("Login succeeded but no token was returned.")
            return

        print("Authentication successful.")

        # Keep JWT in memory; never print it.
        session.headers.update({
            "Authorization": f"Bearer {token}"
        })

        dashboard_response = session.get(
            f"{BASE_URL}/identity/api/v2/user/dashboard",
            timeout=15
        )

        print(
            "Dashboard status:",
            dashboard_response.status_code
        )

        if dashboard_response.status_code == 200:
            print("Dashboard access successful.")
            print("Response is valid JSON:",
                  isinstance(dashboard_response.json(), dict))
        else:
            print("Dashboard access unsuccessful.")
        
        vehicles_response = session.get(
            f"{BASE_URL}/identity/api/v2/vehicle/vehicles",
            timeout=15
        )

        print("Vehicles status:", vehicles_response.status_code)

        if vehicles_response.status_code == 200:
            vehicles = vehicles_response.json()

            for vehicle in vehicles:
                print("Vehicle UUID:", vehicle.get("uuid"))
        else:
            print("Vehicle listing unsuccessful.")
        
        user_b_vehicle_uuid = "7919479f-8f90-4f07-a786-07721111eafd"

        location_response = session.get(
            f"{BASE_URL}/identity/api/v2/vehicle/{user_b_vehicle_uuid}/location",
            timeout=15
        )

        print("Cross-user location status:", location_response.status_code)
        
        if location_response.status_code == 200:
            location_data = location_response.json()

            print(
                "Returned vehicle UUID:",
                location_data.get("carId", "Field not found")
            )

            print(
                "Response JSON fields:",
                list(location_data.keys())
            )
    

    except requests.RequestException as exc:
        print("HTTP request failed:", type(exc).__name__)
    finally:
        session.close()


if __name__ == "__main__":
    main()
