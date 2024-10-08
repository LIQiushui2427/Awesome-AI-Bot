# %%
import requests
import json
from datetime import datetime, timedelta
import pandas as pd

API_PREFIX = "https://datalouder.com"
LOGIN = "yijian2427@gmail.com"  # Put your email or phone number here
PASSWORD = "Qq6059160"  # Put your password here.

API_TOKEN = "706812bd-d103-4a1e-80fd-061cda9525c5"


def get_token(login, password) -> str:
    r = requests.post(
        f"{API_PREFIX}/apiv2/user/login", json={"login": login, "password": password}
    )
    return r.json()["token"]


def make_auth_headers(login, token):
    content = {"email": login, "tokenKey": token}
    return {
        "Authorization": f"DATALOUDER {json.dumps(content)}",
        "Datalouder-Client-Build": "",
        "Datalouder-Client-Version": "100000",
    }


# %%
# API_TOKEN = get_token(LOGIN, PASSWORD)
# %%
def query_price(ticker, end_date=None):
    params = {"stockCode": ticker, "loadAllPrices": "true"}
    if end_date:
        params["endDate"] = end_date
    r = requests.get(
        f"{API_PREFIX}/api/dailyprice/lookup",
        params=params,
        headers=make_auth_headers(LOGIN, API_TOKEN),
    )
    prices = r.json()["prices"]
    if prices:
        prices = [*query_price(ticker, prices[0]["date"]), *prices]
    return prices


def query_price(partial_stock_id, end_date=None):
    olhvc_key = f"olhcv:{partial_stock_id}"
    ccass_key = f"ccass:{partial_stock_id}"
    mktcap_key = f"marketcap:{partial_stock_id}"
    queries = (
        [olhvc_key, mktcap_key, ccass_key]
        if partial_stock_id.startswith("HK")
        else [olhvc_key, mktcap_key]
    )
    params = {"data": ",".join(queries), "unit": "DAY"}
    if end_date:
        params["lastDate"] = end_date
    r = requests.get(
        f"{API_PREFIX}/apiv2/timeseries",
        params=params,
        headers=make_auth_headers(LOGIN, API_TOKEN),
    )
    prices = [
        {
            "dt": x["dt"],
            **x.get(olhvc_key, {}),
            **x.get(ccass_key, {}),
            **x.get(mktcap_key, {}),
        }
        for x in r.json()["data"]
    ]
    if prices:
        prices = [*query_price(partial_stock_id, prices[0]["dt"]), *prices]
    return prices


def query_market_breadth(api_token, partial_index_id, end_date=None):
    """
    Possible partial_index_id:
    - "HK#HSI"
    - "US#IXIC"
    - "US#GSPC"
    - "US#DJI"
    """
    data_key = f"MARKETBREADTH:{partial_index_id}"
    params = {"data": data_key, "unit": "DAY"}
    if end_date:
        params["lastDate"] = end_date
    r = requests.get(
        f"{API_PREFIX}/apiv2/timeseries",
        params=params,
        headers=make_auth_headers(LOGIN, api_token),
    )

    data = [
        {
            "dt": x["dt"],
            **{
                f"{key}-{k}": v
                for (key, value) in x.get(data_key, {}).items()
                for (k, v) in value.items()
            },
        }
        for x in r.json()["data"]
    ]
    if data:
        data = [
            *query_market_breadth(
                api_token=api_token,
                partial_index_id=partial_index_id,
                end_date=data[0]["dt"],
            ),
            *data,
        ]
    return data


if __name__ == "__main__":
    api_token = get_token(LOGIN, PASSWORD)
    print("api_token: " + api_token)
    data = query_market_breadth(api_token=api_token, partial_index_id="HK#HSI")
    df = pd.DataFrame(data)
    print(df.columns)
    print(df)
    df.set_index("dt", inplace=True)
    df.to_csv(f"HK#HSI_market_breadth.csv", index=False)
    # headers = {
    #     'email': 'yijian2427@gmail.com',
    #     'tokenKey': '8d9f2b23-7184-4e64-b274-6dadeea152b2'
    # }
    # make_auth_headers(LOGIN, API_TOKEN)
