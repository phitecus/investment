import market_timing
import functions
import numpy as np
import os
from random import randint
import csv
import time
tickers = market_timing.tickers
price = market_timing.price
timestamp = market_timing.timestamp
market_timing = market_timing.market_timing

# Set the directory
directory = "Optimization/"
if not os.path.exists(directory):
    os.makedirs(directory)

iteration = 500
header = ["RSI Period", "RSI Low", "RSI High", "MACD Fast", "MACD Slow", 'MACD Sig', "EMA Slow", "EMA Fast", "Sharpe", "Annual(%)"]


def optimization():
    # Loop Through Stocks
    for i in range(len(tickers)):
        stock = tickers[i]
        list_price = price[:, i+1]
        # Change all the values to float
        list_price = np.asfarray(list_price, float)
        boundary = len(list_price)

        # Criteria_RSI
        RSI_Low = randint(10, 50)  # percentage (10% - 50%)
        RSI_High = randint(51, 90)  # percentage (51% - 90%)
        # Criteria_MACD (signal line)
        MACD_Sig = randint(2, 20)  # between 2 and 20
        # Criteria_EMA (short-term ema)
        EMA_Fast = randint(5, 80)  # between 5 and 80

        # RSI
        RSI_period = randint(4, 60)  # between 4 and 60
        # MACD
        new_boundary = boundary - MACD_Sig
        MACD_Fast = randint(MACD_Sig, min(300, new_boundary))  # bigger than MACD_Sig
        # bigger than MACD_Sig and MACD_Fast
        MACD_Slow = randint(max(MACD_Fast+1, int(MACD_Fast*1.3)), min(MACD_Fast*10, 1000, new_boundary))
        # EMA
        EMA_Slow = randint(max(EMA_Fast+1, int(EMA_Fast*1.3)), min(EMA_Fast*20, 800, boundary))  # bigger than EMA_Fast

        result = market_timing(stock, list_price, RSI_period, MACD_Fast, MACD_Slow, EMA_Slow, RSI_Low, RSI_High,
                               MACD_Sig, EMA_Fast)
        sharpe = result[0]
        cumulative = result[1]

        # Import result matrix
        title = directory + functions.change_title(stock) + ' optimization.csv'
        # If it does not exist, we create result matrix
        if not os.path.exists(title):
            # Make Result Matrix
            result_matrix = [[0 for x in range(len(header))] for y in range(1)]
            firstrow = result_matrix[0]
            # Fill the headers
            for col in range(len(firstrow)):
                firstrow[col] = header[col]
        # If file exists, we import the result matrix
        else:
            with open(title, "r") as f:
                reader = csv.reader(f)
                result_matrix = [rows for rows in reader]

        new = [RSI_period, RSI_Low, RSI_High, MACD_Fast, MACD_Slow, MACD_Sig, EMA_Slow, EMA_Fast, sharpe, cumulative]
        result_matrix.append(new)

        # Write csv file of result matrix inside result folder
        with open(title, "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            for val in result_matrix:
                writer.writerow(val)


print("Start")
for count in range(iteration):
    start = time.time()
    optimization()
    end = time.time()
    print("Trial", count + 1, ":", round(end-start, 4), "seconds")

