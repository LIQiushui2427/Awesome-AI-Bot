
# Awesome-backtesting

This is a curated backtesting framework for quantitative trading for futures and options taking advantage of [CoT](https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm) report from [CFTC](https://www.cftc.gov/).
Users can make composite strategies by combining different strategies offered here.
They can test their strategies on historical data and see how they perform.
This project is built on [backtesting.py](https://kernc.github.io/backtesting.py/).

## Table 
## Quick Start

- [Awesome-backtesting](#awesome-backtesting)
  - [Table](#table)
  - [Quick Start](#quick-start)
    - [Step 0: Install Dependencies](#step-0-install-dependencies)
    - [Step 1: Download data](#step-1-download-data)

### Step 0: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 1: Download data

Supported data:(To be updated)

|  Future-only  | Link |
|  ----  | ----  |
| Disaggregated Commitments of Traders  | [2022-txt](https://www.cftc.gov/files/dea/history/fut_fin_txt_2022.zip)|
| Traders in Financial Futures (TFF)  | [2022-txt](https://www.cftc.gov/files/dea/history/fut_fin_txt_2022.zip)|

| Future-Options-Combined  | Link |
|  ----  | ----  |
| Disaggregated Commitments of Traders  | [2022-txt](https://www.cftc.gov/files/dea/history/fut_fin_txt_2022.zip)|
| Traders in Financial Futures (TFF)  | [2022-txt](https://www.cftc.gov/files/dea/history/fut_fin_txt_2022.zip)|

Download the data, unzip it and put it in the `data` folder.
