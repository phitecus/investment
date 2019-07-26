import functions
import csv
import os
import numpy as np
# Functions
macd = functions.get_macd  # 3 random variables
ema = functions.get_ema  # 2 random variables
rsi = functions.get_rsi  # 1 random variable

timestamp = []
# Import prices of each stock
with open('prices.csv') as g:
    reader = csv.reader(g)
    price = [rows for rows in reader]
    price = functions.replace_na(price)
    tickers = price[0][1:]
    price = price[2:]
    for i in range(len(price)):
        timestamp.append(price[i][0])

# Set the directory
directory = "Market Timing Result/"
if not os.path.exists(directory):
    os.makedirs(directory)

price = np.array(price)
trading_cost = 0.2  # percentage


def market_timing(stock, pricelist, RSI_period, MACD_Fast, MACD_Slow, EMA_Slow, RSI_Low, RSI_High, MACD_Sig, EMA_Fast):
    header = ["Date", "Price", "RSI", "MACD", "EMA", "RSI signal", "MACD Signal", "EMA Signal", "Signal In", "Signal Out", "Position", "Trading Cost (%)", "Return(%)", "Cumulative(%)"]
    price_matrix = [[0 for x in range(len(header))] for y in range(len(timestamp)+1)]
    delete = functions.findzero(pricelist)
    list_return = pricelist[:delete].tolist()
    pricelist = pricelist[delete:]
    RSI = rsi(pricelist, RSI_period)
    MACD = macd(pricelist, MACD_Fast, MACD_Slow)
    signal_line = ema(MACD, MACD_Sig)
    EMA = ema(pricelist, EMA_Slow)
    short_ema = ema(pricelist, EMA_Fast)
    # Count index
    index_macd = MACD_Slow - 1 + delete
    index_signal_line = MACD_Slow - 1 + MACD_Sig - 1 + delete
    index_ema = EMA_Slow - 1 + delete
    index_short_ema = EMA_Fast - 1 + delete
    index_rsi = RSI_period + delete
    index_signal = max(index_rsi, index_signal_line, index_ema)
    numberofday = len(price_matrix) - index_signal - 1
    # Fill the headers
    firstrow = price_matrix[0]
    for integer in range(len(firstrow)):
        firstrow[integer] = header[integer]

    # Fill the rest of values
    position = 0
    cumulative = 0
    portfolio = 1
    for integer in range(len(price_matrix)-1):
        price_matrix[integer+1][0] = timestamp[integer]
        if integer >= delete:
            signal_rsi = 0
            signal_macd = 0
            signal_ema = 0
            price_matrix[integer + 1][1] = pricelist[integer-delete]
            # RSI
            if integer >= index_rsi:
                value_rsi = RSI[integer-index_rsi]
                price_matrix[integer + 1][2] = value_rsi
                if RSI_Low < value_rsi < RSI_High:
                    signal_rsi = 1
                else:
                    signal_rsi = 0
                price_matrix[integer + 1][5] = signal_rsi

            # MACD
            if integer >= index_macd:
                value_macd = MACD[integer-index_macd]
                price_matrix[integer + 1][3] = value_macd
                if integer >= index_signal_line:
                    value_signal_line = signal_line[integer-index_signal_line]
                    if value_macd > value_signal_line:
                        signal_macd = 1
                        price_matrix[integer + 1][6] = signal_macd

            # EMA
            if integer >= index_short_ema:
                value_short_ema = short_ema[integer-index_short_ema]
                if integer >= index_ema:
                    value_ema = EMA[integer-index_ema]
                    price_matrix[integer + 1][4] = value_ema
                    if value_ema < value_short_ema:
                        signal_ema = 1
                        price_matrix[integer + 1][7] = signal_ema

            # Signals and Positions
            in_signal = 0
            out_signal = 0
            if integer >= index_signal:
                # Signal In
                if signal_rsi == 1 and signal_macd == 1 and signal_ema == 1:
                    in_signal = 1
                    price_matrix[integer + 1][8] = in_signal

                # Signal Out
                if signal_ema == 0:
                    out_signal = 1
                    price_matrix[integer + 1][9] = out_signal

                # Position
                if in_signal == 1 and out_signal == 1:
                    position = 0
                elif in_signal == 1 and out_signal == 0:
                    position = 1
                elif in_signal == 0 and out_signal == 1:
                    position = 0
                price_matrix[integer + 1][10] = position

                # Trading Cost
                if price_matrix[integer + 1][10] != price_matrix[integer][10]:
                    price_matrix[integer + 1][11] = trading_cost

            # Return
            if integer > delete:
                gain = functions.percentageChange(pricelist[integer - delete - 1], pricelist[integer - delete])
                gain *= int(price_matrix[integer][10])  # previous period's position
                profit = gain - price_matrix[integer + 1][11]  # return - cost
                price_matrix[integer + 1][12] = gain
                list_return.append(profit)
                # Cumulative Return
                portfolio *= (1 + profit / 100)
                cumulative = (portfolio-1)*100
                price_matrix[integer + 1][13] = cumulative

    # Annualized Sharpe and Return
    if numberofday != 0:
        list_return = list_return[-numberofday:]
        cumulative = cumulative*252/numberofday
        sharpe = functions.Sharpe(list_return)
    else:
        cumulative = 0
        sharpe = 0

    # Write csv file of result matrix inside result folder
    title = directory + functions.change_title(stock)
    with open(title + ' technical indicators.csv', "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in price_matrix:
            writer.writerow(val)

    return [sharpe, cumulative]
