import functions
import csv
import numpy as np
directory = "Market Timing Result/"
trading_cost = 0.2  # percentage


def combined_signal(stock, weight_scheme, stock_index):
    header = ["Date", "Price", "Weight Signal", "Signal In", "Combined Signal", "Signal Out", "Position", "Trading Cost", "Return(%)", "Cumulative(%)"]
    title = directory + functions.change_title(stock)
    with open(title + ' technical indicators.csv') as f:
        reader = csv.reader(f)
        original = [rows for rows in reader]
    original = np.array(original)
    signal_matrix = [[0 for x in range(len(header))] for y in range(len(original))]
    signal_matrix[0] = header
    signal_matrix = np.array(signal_matrix)
    signal_matrix[:, 0][1:] = original[:, 0][1:]  # Date
    signal_matrix[:, 1][1:] = original[:, 1][1:]  # Price
    signal_matrix[:, 3][1:] = original[:, 8][1:]  # signal in
    signal_matrix[:, 5][1:] = original[:, 9][1:]  # signal out
    # Import Weight Scheme
    with open('Stock Picking Result/' + weight_scheme + '.csv') as g:
        reader = csv.reader(g)
        weight = [rows for rows in reader]
    weight = np.array(weight)
    time_weight = weight[:, 0][1:]
    timestamp = signal_matrix[:, 0][1:]
    position = 0
    portfolio = 1
    for time in range(len(timestamp)):
        answer = 0
        month = int(timestamp[time].split("/")[1])
        year = int(timestamp[time].split("/")[2])
        # Q-2 quarter
        if month < 4:  # Q1
            year -= 1
            quarter = "Q2"
        elif month < 10:  # Q2 and Q3
            year -= 1
            quarter = "Q4"
        else:  # Q4
            quarter = "Q2"
        date = str(year) + quarter
        # Search for index for weights
        for index in range(len(time_weight)):
            if date == time_weight[index]:
                answer = index
        if answer == 0:
            stockweight = 0
        else:
            stockweight = float(weight[answer + 1][stock_index + 1]) / 100
        # Fill in the weight signal in second column
        if stockweight == 0:
            signal_matrix[time + 1][2] = 0
            weight_signal = 0
        else:
            signal_matrix[time + 1][2] = 1
            weight_signal = 1
        # Combined signal = 1 if both signal is 1
        technical_signal = int(signal_matrix[time + 1][3])
        if weight_signal == 1 and technical_signal == 1:
            in_signal = 1
        else:
            in_signal = 0
        signal_matrix[time + 1][4] = in_signal  # Combined signal
        # Position
        out_signal = int(signal_matrix[time + 1][5])
        if in_signal == 1 and out_signal == 1:
            position = 0
        elif in_signal == 1 and out_signal == 0:
            position = 1
        elif in_signal == 0 and out_signal == 1:
            position = 0
        if weight_signal == 0:
            position = 0
        signal_matrix[time + 1][6] = position  # Position
        # Trading Cost
        if time != 0 and signal_matrix[time + 1][6] != signal_matrix[time][6]:
            signal_matrix[time + 1][7] = trading_cost  # Trading Cost
        # Return
        if time != 0:
            gain = functions.percentageChange(float(signal_matrix[time][1]), float(signal_matrix[time + 1][1]))
            gain *= int(signal_matrix[time][6])  # previous period's position
            profit = gain - float(signal_matrix[time + 1][7])  # return - cost
            signal_matrix[time + 1][8] = gain  # return
            # Cumulative Return
            portfolio *= (1 + profit / 100)
            cumulative = (portfolio - 1) * 100
            signal_matrix[time + 1][9] = cumulative  # cumulative return

    # Write csv file of result matrix inside result folder
    title = directory + functions.change_title(stock)
    with open(title + ' combined signal ' + weight_scheme + '.csv', "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in signal_matrix:
            writer.writerow(val)


#tickers = market_timing.tickers
#for stock in range(len(tickers)):
#    stocks = functions.change_title(tickers[stock])
#    print(stocks)
#    combined_signal(tickers[stock], 'weight 1', stock)
#    combined_signal(tickers[stock], 'weight 2', stock)
