import asyncio

import httpx


async def make_request(endpoint, params=None):
    url = f"http://localhost:6677/{endpoint}"
    async with httpx.AsyncClient(timeout=20) as client:
        if params:
            response = await client.post(url, params=params)
        else:
            response = await client.get(url)

        if response.status_code == 200:
            print(f"Response from {endpoint}: {response.json()}")
        else:
            print(f"Error from {endpoint}: {response.status_code}")
            print("Response content:", response.text)


async def main():
    chat_params = {"msg": "How much does a beer cost?"}
    tasks = [make_request("get-all-ids/"), make_request("chat/", params=chat_params)]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
