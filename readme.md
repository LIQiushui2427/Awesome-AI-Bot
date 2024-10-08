
# Awesome-AI-Trader

This is a curated automated AI app for quantitative trading for stocks, futures and options. intergrated data fetching before it, and backtesting after it. For data, it taks advantage of [Yahoo finance](https://finance.yahoo.com/) (For daily price data) and  [CFTC](https://www.cftc.gov/) (for Commitment of Traders data).

- [Awesome-AI-Trader](#awesome-ai-trader)
  - [Project proposal / design](#project-proposal--design)
  - [Pipeline](#pipeline)
  - [To do list](#to-do-list)
  - [Local run: Quick Start](#local-run-quick-start)
    - [Requirement](#requirement)
    - [Start a demo](#start-a-demo)
  - [License](#license)

## Project proposal / design

This project is trying to automate the data manipulation, AI tuning for quantitative trading, and provide good user experience like visualization. Its objective is to give a good buy/sell signal to user.

For details, please find the Project proposal / design in  [documentation](./documentation.pdf).

## Pipeline

1. Data fetching: Fetch data from Yahoo finance, CFTC, and other sources.
2. Data manipulation: Clean and manipulate data: fill missing data, normalize data, etc.
3. Feature extraction: Extract features from data, like moving average, RSI, etc.
4. Preprocessing: Use MIC, Ramdom Forest, etc to get the best features for AI model.
5. Train AI model: Use LSTM, CNN, etc to predict the price.
6. Backtesting: Backtest the AI model with historical data.
7. Visualization: Visualize the backtesting result.
8. User interface: Provide a good user interface for user to interact with the app.


## To do list

- [x] Adpot constant strategy when training AI.
- [ ] Compute whole market infomation and add this feed into every single stock data.

## Local run: Quick Start

Though This project is oriented to serve as some beckend for a potential application in the future, user can still run and test it locally.

### Requirement

User is expected to have [docker](https://www.docker.com/) installed in their computer.

### Start a demo

After you download this project, open a terminal (git.bash if you are using Windows). Start docker service, if not, and run the following:

```bash
./deploy.sh
```

And this app will be built, start ,and work in the terminal.

## License

[MIT](./LICENSE)