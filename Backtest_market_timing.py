import market_timing
import functions
import numpy as np
import pandas as pd
import os
import csv
import time
import matplotlib.pyplot as plt
import weight1
import weight2
import pylab as pl
import scipy.stats as stats
import combined_signal

# Variables
initial_amount = 1000000
year_input = 2015
optimization_count = 500  # Avoid over-fitting

tickers = market_timing.tickers
price = market_timing.price
timestamp = market_timing.timestamp
market_timing = market_timing.market_timing
combined_signal = combined_signal.combined_signal
# Initialization
market = "HSCI"
date_index = []
price_index = []
store_index = []  # when quarter changes
stock_pnl_pure = []
stats_matrix = [[0 for x in range(6)] for y in range(1)]
# Functions
average = functions.findaverage
SD = functions.findSD
maxconsecutive = functions.MaxConsecutive
winrate = functions.WinRate
sharpe = functions.Sharpe
head_stats = [str(year_input), "Cumulative Return (%)", "Annualized Return (%)", "Annualized Volatility (%)", "Sharpe Ratio", "Max Drawdown (%)"]
for i in range(len(stats_matrix[0])):
    stats_matrix[0][i] = head_stats[i]

# Import market index
with open(market+'.csv') as f:
    reader = csv.reader(f)
    index = [rows for rows in reader]
    for i in range(len(index)):
        date_index.append(index[i][0])
        price_index.append(float(index[i][1]))

# Find the timestamp of designated year and delete previous ones
if year_input <= 2000 or year_input >= 2018:
    print("Use Default Year")
    year_input = 2000

print("\n")
print(market, "MARKET TIMING", "start from", year_input)

# initialize result matrix
result_matrix = [[0 for x in range(len(tickers)+8)] for y in range(len(timestamp)+1)]  # Pure Market Trending
result_matrix1 = [[0 for x in range(len(tickers)+3)] for y in range(len(timestamp)+1)]  # Combined Strategy
w1_matrix = [[0 for x in range(len(tickers)+1)] for y in range(len(timestamp)+1)]
w2_matrix = [[0 for x in range(len(tickers)+1)] for y in range(len(timestamp)+1)]
w3_matrix = [[0 for x in range(len(tickers)+1)] for y in range(len(timestamp)+1)]
# Fill the headers
firstrow = result_matrix[0]
firstrow1 = result_matrix1[0]
firstrow_w1 = w1_matrix[0]
firstrow_w2 = w2_matrix[0]
firstrow_w3 = w3_matrix[0]
for integer in range(len(firstrow)):
    if integer == 0:
        firstrow[integer] = ""
        firstrow1[integer] = ""
        firstrow_w1[integer] = ""
        firstrow_w2[integer] = ""
        firstrow_w3[integer] = ""
    elif integer <= len(tickers):
        firstrow[integer] = tickers[integer - 1]
        firstrow1[integer] = tickers[integer - 1]
        firstrow_w1[integer] = tickers[integer - 1]
        firstrow_w2[integer] = tickers[integer - 1]
        firstrow_w3[integer] = tickers[integer - 1]
    elif integer == len(tickers)+1:
        firstrow[integer] = "Return before trading cost"
        firstrow1[integer] = "Portfolio"
    elif integer == len(tickers) + 2:
        firstrow[integer] = "Trading Cost"
        firstrow1[integer] = "Daily Return"
    elif integer == len(tickers) + 3:
        firstrow[integer] = "Daily Return"
    elif integer == len(tickers) + 4:
        firstrow[integer] = "Cumulative Return"
    elif integer == len(tickers) + 5:
        firstrow[integer] = "Weight Scheme 1"
    elif integer == len(tickers) + 6:
        firstrow[integer] = "Weight Scheme 2"
    elif integer == len(tickers) + 7:
        firstrow[integer] = "Optimized Weight"
result_matrix = np.array(result_matrix)
result_matrix[:, 0][1:] = timestamp
result_matrix1 = np.array(result_matrix1)
result_matrix1[:, 0][1:] = timestamp
column_cost = np.array(result_matrix[:, -6][1:])
column_cost = np.asfarray(column_cost, float)
column_return = np.array(result_matrix[:, -7][1:])
column_return = np.asfarray(column_return, float)
print("Start Optimization...")
start = time.time()
# Loop for stocks
for i in range(len(tickers)):
    stock = functions.change_title(tickers[i])
    print(stock)
    list_price = price[:, i + 1]
    # Change all the values to float
    list_price = np.asfarray(list_price, float)
    # Set the directory
    directory = "Optimization/"

    # Import Optimization Result
    with open(directory + stock + ' optimization.csv') as f:
        reader = csv.reader(f)
        results = [rows for rows in reader]
        results = results[1:optimization_count]  # Remove headers
    results = np.array(results)
    results = np.asfarray(results, float)  # Change all the values to float
    annual_return = results[:, -1]  # Last Column = Annual Return (%)
    maximum_return = functions.findMax(annual_return)
    # Find the index of maximum return
    index = functions.findMaxIndex(annual_return)
    # Technical Variables
    RSI_period = int(results[:, 0][index])
    RSI_Low = int(results[:, 1][index])
    RSI_High = int(results[:, 2][index])
    MACD_Fast = int(results[:, 3][index])
    MACD_Slow = int(results[:, 4][index])
    MACD_Sig = int(results[:, 5][index])
    EMA_Slow = int(results[:, 6][index])
    EMA_Fast = int(results[:, 7][index])
    #market_timing(stock, list_price, RSI_period, MACD_Fast, MACD_Slow, EMA_Slow, RSI_Low, RSI_High, MACD_Sig, EMA_Fast)
    #combined_signal(stock, 'weight 1', i)
    #combined_signal(stock, 'weight 2', i)
    #combined_signal(stock, 'weight optimized', i)
    # Import results after market timing
    directory = "Market Timing Result/"
    title = directory + functions.change_title(stock)
    # Pure Market Timing
    with open(title + ' technical indicators.csv') as g:
        reader = csv.reader(g)
        results = [rows for rows in reader]
    # Weight Scheme 1
    with open(title + ' combined signal weight 1.csv') as h:
        reader = csv.reader(h)
        results_w1 = [rows for rows in reader]
    w1_matrix = np.array(w1_matrix)
    results_w1 = np.array(results_w1)
    positive_w1 = results_w1[:, -2][1:]
    positive_w1 = np.asfarray(positive_w1, float)
    negative_w1 = results_w1[:, -3][1:]
    negative_w1 = np.asfarray(negative_w1, float)
    w1_matrix[:, i + 1][1:] = positive_w1 - negative_w1
    # Weight Scheme 2
    with open(title + ' combined signal weight 2.csv') as j:
        reader = csv.reader(j)
        results_w2 = [rows for rows in reader]
    w2_matrix = np.array(w2_matrix)
    results_w2 = np.array(results_w2)
    positive_w2 = results_w2[:, -2][1:]
    positive_w2 = np.asfarray(positive_w2, float)
    negative_w2 = results_w2[:, -3][1:]
    negative_w2 = np.asfarray(negative_w2, float)
    w2_matrix[:, i + 1][1:] = positive_w2 - negative_w2
    # Weight Optimized
    with open(title + ' combined signal weight optimized.csv') as k:
        reader = csv.reader(k)
        results_w3 = [rows for rows in reader]
    w3_matrix = np.array(w3_matrix)
    results_w3 = np.array(results_w3)
    positive_w3 = results_w3[:, -2][1:]
    positive_w3 = np.asfarray(positive_w3, float)
    negative_w3 = results_w3[:, -3][1:]
    negative_w3 = np.asfarray(negative_w3, float)
    w3_matrix[:, i + 1][1:] = positive_w3 - negative_w3

    results = results[1:]  # Remove headers
    results = np.array(results)
    daily_return = results[:, -2]
    trading_cost = np.array(results[:, -3])
    result_matrix[:, i+1][1:] = daily_return
    daily_return = np.array(daily_return)
    daily_return = np.asfarray(daily_return, float)
    trading_cost = np.asfarray(trading_cost, float)
    column_return += daily_return
    column_cost += trading_cost

result_matrix[:, -7][1:] = column_return / len(tickers)
result_matrix[:, -6][1:] = column_cost / len(tickers)
daily_profit = (column_return - column_cost) / len(tickers)
result_matrix[:, -5][1:] = daily_profit
cumulative_pure = functions.cumulative_list(daily_profit)
result_matrix[:, -4][1:] = cumulative_pure
end = time.time()
timing = end - start
print(int(timing/60), " minutes and", round(timing-60*int(timing/60), 4), " seconds")

# Weight 1
weight_scheme_1 = weight1.weightMatrix
# Weight 2
weight_scheme_2 = weight2.weightMatrix
# Optimized Weight
with open('Stock Picking Result/' + 'weight optimized.csv') as i:
    reader = csv.reader(i)
    weight_optimized = [rows for rows in reader]

directory = "Market Timing Performance/"
if not os.path.exists(directory):
    os.makedirs(directory)


def trend_following(weight_scheme, number):
    quarter_list = []
    # Weight Scheme 1: Third Last Column
    # Weight Scheme 2: Second Last Column
    # Weight Optimized: Last Column
    column_return = np.array(result_matrix[:, number - 4][1:])
    column_return = np.asfarray(column_return, float)
    column_return[0] = initial_amount

    # Loop through time index
    cash_amount = 0
    for index in range(len(timestamp)):
        bool_quarter = False
        answer = 0  # search for index for weights
        month = int(timestamp[index].split("/")[1])
        year = int(timestamp[index].split("/")[2])
        if month < 7:
            quarter = "Q2"
        else:
            quarter = "Q4"
        current = str(year) + quarter

        # Q-2 quarter
        if month < 4:
            year -= 1
            quarter = "Q2"
        elif month < 7:
            year -= 1
            quarter = "Q4"
        elif month < 10:
            year -= 1
            quarter = "Q4"
        else:
            quarter = "Q2"
        date = str(year) + quarter

        # Check if quarter is different from previous day's quarter
        quarter_list.append(current)
        if number == 1:
            if index == 0:
                store_index.append([index, current])
            else:
                if quarter_list[index] != quarter_list[index - 1]:
                    store_index.append([index, current])
        if (index == 0) or (index != 0 and quarter_list[index] != quarter_list[index - 1]):
            bool_quarter = True

        # Search for index for weights
        for index2 in range(len(weight_scheme)):
            if date == weight_scheme[index2][0]:
                answer = index2

        # Loop through stocks
        aggregate = 0
        cash_hand = 100
        if number == 1:
            w_return = w1_matrix
        elif number == 2:
            w_return = w2_matrix
        else:
            w_return = w3_matrix
        for stock in range(len(tickers)):
            # Import results after market timing
            gain = float(w_return[index+1][stock+1])
            if answer == 0:
                stockweight = 0
            else:
                w = float(weight_scheme[answer][stock + 1])
                stockweight = w / 100
            if bool_quarter:  # When quarter changes, change the initial amount and re-balance
                if index == 0:
                    value = initial_amount * stockweight
                else:
                    value = column_return[index-1] * stockweight
                    cash_hand -= w
            else:  # When quarter doesn't change
                value = float(result_matrix1[index][stock+1]) * (1 + gain / 100)
            result_matrix1[index+1][stock+1] = value  # Fill in the money allocation instead of percentage change
            aggregate += value
        # Calculate cash in hand when remaining weight is not zero
        if bool_quarter and cash_hand < 100:
            cash_amount = cash_hand * column_return[index-1] / 100
        if bool_quarter and cash_hand == 100:
            cash_amount = 0
        if aggregate == 0:
            aggregate = column_return[index-1]
            if column_return[index-1] == 0 or index == 0:
                aggregate = initial_amount
        aggregate += cash_amount
        column_return[index] = aggregate

    result_matrix1[:, -2][1:] = column_return  # Portfolio
    list_return = functions.percentagelist(column_return)
    result_matrix1[:, -1][2:] = list_return  # Daily Return
    cumulative = functions.cumulative_list(list_return)
    result_matrix[:, number - 4][2:] = cumulative  # Market Timing Strategy.csv

    if number == 1:
        # Write csv file of result matrix inside result folder
        with open(directory + 'Combined Strategy 1.csv', "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            for val in result_matrix1:
                writer.writerow(val)
    elif number == 2:
        # Write csv file of result matrix inside result folder
        with open(directory + 'Combined Strategy 2.csv', "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            for val in result_matrix1:
                writer.writerow(val)
    else:
        # Write csv file of result matrix inside result folder
        with open(directory + 'Combined Strategy Optimized Weight.csv', "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            for val in result_matrix1:
                writer.writerow(val)


# Fill in the results of each weight scheme
trend_following(weight_scheme_1, 1)
trend_following(weight_scheme_2, 2)
trend_following(weight_optimized, 3)
cumulative_weight1 = np.asfarray(result_matrix[:,  -3][1:], float)
cumulative_weight2 = np.asfarray(result_matrix[:,  -2][1:], float)
cumulative_weight_optimized = np.asfarray(result_matrix[:,  -1][1:], float)
portfolio_weight1 = functions.portfolio_list(cumulative_weight1, initial_amount)
portfolio_weight2 = functions.portfolio_list(cumulative_weight2, initial_amount)
portfolio_weight_optimized = functions.portfolio_list(cumulative_weight_optimized, initial_amount)
portfolio_pure = functions.portfolio_list(cumulative_pure, initial_amount)

# Write csv file of result matrix inside result folder
with open(directory + 'Market Timing Strategy.csv', "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in result_matrix:
        writer.writerow(val)

# Pure Market Timing Normal Graph
for stock in range(len(tickers)):
    pnl = np.asfarray(result_matrix[:,  stock+1][1:], float)
    stock_pnl_pure.append(pnl)
# PnL
stock_pnl = functions.daily_return_to_annual_pnl(stock_pnl_pure)
fig = pl.figure()
h = sorted(stock_pnl)
fit = stats.norm.pdf(h, np.mean(h), np.std(h))
pl.title("Pure Market Timing Annual P&L Distribution")
pl.plot(h, fit, color='xkcd:grey')
pl.hist(h, density=True, color='xkcd:beige')  # use this to draw histogram of your data
pl.xlabel('Stock P&L (%)')
fig.savefig(directory + "Pure Market Timing Annual P&L Distribution.png")
# Sharpe Ratio
stock_sharpe = functions.daily_return_to_sharpe(stock_pnl_pure)
fig = pl.figure()
i = sorted(stock_sharpe)
fit = stats.norm.pdf(i, np.mean(i), np.std(i))
pl.title("Pure Market Timing Sharpe Distribution")
pl.plot(i, fit, color='xkcd:grey')
pl.hist(i, density=True, color='xkcd:beige')  # use this to draw histogram of your data
pl.xlabel('Sharpe Ratio')
fig.savefig(directory + "Pure Market Timing Sharpe Distribution.png")
# Daily PnL
#daily_pnl_pure = np.asfarray(result_matrix[:, -4][1:], float)
#fig = pl.figure()
#j = sorted(daily_pnl_pure)
#fit = stats.norm.pdf(j, np.mean(j), np.std(j))
#pl.title("Pure Market Timing Daily P&L Distribution")
#pl.plot(j, fit, color='xkcd:grey')
#pl.hist(j, density=True, color='xkcd:beige')  # use this to draw histogram of your data
#pl.xlabel('Stock P&L (%)')
#fig.savefig(directory + "Pure Market Timing daily P&L Distribution.png")

# Import Sector Selection
df = pd.read_csv('Sector Selection Result/' + 'result matrix.csv', low_memory=False)  # does not include header
df = np.array(df)[1:]  # start from Jan 04, 2000

##################################################
#import Backtest_sector_selection
#print("Plot Cumulative Graphs...")
#print("\n")
#start = time.time()
# Draw Cumulative Graphs for each stock and save it in a folder
#if not os.path.exists(directory + 'stocks/'):
#    os.makedirs(directory + 'stocks/')
#x_axis = timestamp
# 2010 Q3 - Q4 [2740:2870]
# 2012 Q3 - Q4 [3262:3391]
# 2013 Q3 - Q4 [3522:3652]
# 2015 Q1 - Q2 [3915:4042]
# 2017 Q3 - Q4 [4567:4695]
#x_axis = x_axis[4567:4695]
#for stock in range(len(tickers)):
#    # Pure Market Trending
#    y = functions.portfolio_list(functions.cumulative_list(stock_pnl_pure[stock]), initial_amount)[:-1]
#    y = y[4567:4695]
#    y = functions.relative(y)
#    # Market Trending + Sector Selection
#    stock_pnl_sector_selection = np.asfarray(df[:, stock + 1], float)
#    boolean = stock_pnl_sector_selection != 0  # If the value is zero, change to zero
#    y1 = stock_pnl_pure[stock] * boolean
#    y1 = functions.portfolio_list(functions.cumulative_list(y1), initial_amount)[:-1]
#    y1 = y1[4567:4695]
#    y1 = functions.relative(y1)
#    # Benchmark
#    price_market = np.asfarray(price[:, stock + 1], float)
#    price_market = price_market[4567:4695]
#    y_market = functions.relative(price_market)
#    fig = plt.figure()  # Create a figure
#    ax = plt.axes()  # Create axis
#    plt.plot(x_axis, y, label='Pure Market Trending')
#    plt.plot(x_axis, y1, label='Market Trending + Sector Selection')
#    plt.plot(x_axis, y_market, label=tickers[stock])
#    ax.xaxis.set_major_locator(plt.MaxNLocator(5))  # Set Maximum number of x-axis values to show
#    fig.autofmt_xdate()  # Rotate values to see more clearly
#    legend = plt.legend(loc='best')
#    title = tickers[stock] + ": Growth of " + str(initial_amount / 1000000) + " million"
#    plt.title(title)
#    plt.ylabel("Cumulative Return")
#    fig.savefig(directory + 'stocks/' + functions.change_title(tickers[stock]) + ".png")
#    plt.close(fig)
#end = time.time()
#timing = end - start
#print(int(timing/60), " minutes and", round(timing-60*int(timing/60), 4), " seconds to plot graphs")
#print("\n")
########################################################
print("Semiannually")
# Calculate Quarterly Return (HK: semiannually)
quarter_matrix = [[0 for x in range(len(store_index))] for y in range(5)]
quarter_matrix[0][0] = "Semiannual Return"
quarter_matrix[1][0] = "Pure Market Timing"
quarter_matrix[2][0] = "Weight Scheme 1 + Market Timing"
quarter_matrix[3][0] = "Weight Scheme 2 + Market Timing"
quarter_matrix[4][0] = "Weight Optimized + Market Timing"
for i in range(len(store_index)-1):
    quarter_matrix[0][i + 1] = store_index[i][1]
    quarter_matrix[1][i + 1] = functions.percentageChange(portfolio_pure[store_index[i][0]],
                                                          portfolio_pure[store_index[i + 1][0]])
    quarter_matrix[2][i + 1] = functions.percentageChange(portfolio_weight1[store_index[i][0]],
                                                          portfolio_weight1[store_index[i + 1][0]])
    quarter_matrix[3][i + 1] = functions.percentageChange(portfolio_weight2[store_index[i][0]],
                                                          portfolio_weight2[store_index[i + 1][0]])
    quarter_matrix[4][i + 1] = functions.percentageChange(portfolio_weight_optimized[store_index[i][0]],
                                                          portfolio_weight_optimized[store_index[i + 1][0]])
# Benchmark Quarter Return: HSCI
quarter_list = []
store_index = []
quarter_benchmark = []
for index in range(len(date_index)):
    bool_quarter = False
    month = int(date_index[index].split("/")[1])
    year = int(date_index[index].split("/")[2])
    if month < 7:
        quarter = "Q2"
    else:
        quarter = "Q4"
    # Check if quarter is different
    quarter_list.append(quarter)
    if index == 0:
        store_index.append(index)
    else:
        if quarter_list[index] != quarter_list[index - 1]:
            store_index.append(index)
for i in range(len(store_index)-1):
    quarter_benchmark.append(functions.percentageChange(price_index[store_index[i]], price_index[store_index[i + 1]]))

# Write csv file of result matrix inside result folder
with open(directory + 'Semiannual Return.csv', "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in quarter_matrix:
        writer.writerow(val)

Quarter = quarter_matrix[0][1:]
Quarter_return_pure = quarter_matrix[1][1:]
Quarter_return1 = quarter_matrix[2][1:]
Quarter_return2 = quarter_matrix[3][1:]
Quarter_return_optimized = quarter_matrix[4][1:]

# Draw quarterly return bar graphs
fig = plt.figure()  # Create a figure
ax = plt.axes()  # Create axis
plt.bar(Quarter, Quarter_return_pure, align='center', alpha=0.5)  # Draw Bar graph
ax.xaxis.set_major_locator(plt.MaxNLocator(8))  # Set Maximum number of x-axis values to show
fig.autofmt_xdate()  # Rotate values to see more clearly
plt.ylabel('%')
plt.title('Pure Market Trending Semiannual Return')
fig.savefig(directory + market + " Pure Market Trending Semiannual Return" + ".png")
# plt.show()

# Weight Scheme 1
fig = plt.figure()  # Create a figure
ax = plt.axes()  # Create axis
plt.bar(Quarter, Quarter_return1, align='center', alpha=0.5)  # Draw Bar graph
ax.xaxis.set_major_locator(plt.MaxNLocator(8))  # Set Maximum number of x-axis values to show
fig.autofmt_xdate()  # Rotate values to see more clearly
plt.ylabel('%')
plt.title('Market Trending + Weight 1')
fig.savefig(directory + market + " Market Trending + Weight 1 Semiannual Return" + ".png")
# plt.show()

# Weight Scheme 2
fig = plt.figure()  # Create a figure
ax = plt.axes()  # Create axis
plt.bar(Quarter, Quarter_return2, align='center', alpha=0.5)  # Draw Bar graph
ax.xaxis.set_major_locator(plt.MaxNLocator(8))  # Set Maximum number of x-axis values to show
fig.autofmt_xdate()  # Rotate values to see more clearly
plt.ylabel('%')
plt.title('Market Trending + Weight 2')
fig.savefig(directory + market + " Market Trending + Weight 2 Semiannual Return" + ".png")
# plt.show()

# Weight Optimized
fig = plt.figure()  # Create a figure
ax = plt.axes()  # Create axis
plt.bar(Quarter, Quarter_return_optimized, align='center', alpha=0.5)  # Draw Bar graph
ax.xaxis.set_major_locator(plt.MaxNLocator(8))  # Set Maximum number of x-axis values to show
fig.autofmt_xdate()  # Rotate values to see more clearly
plt.ylabel('%')
plt.title('Market Trending + Weight Optimized')
fig.savefig(directory + market + " Market Trending + Weight Optimized Semiannual Return" + ".png")
# plt.show()

# Compare bar charts from 2008
Quarter = Quarter[16:]
Quarter_return_pure = Quarter_return_pure[16:]
Quarter_return1 = Quarter_return1[16:]
Quarter_return2 = Quarter_return2[16:]
Quarter_return_optimized = Quarter_return_optimized[16:]
Quarter_return_benchmark = quarter_benchmark[16:]
fig, ax = plt.subplots(figsize=(15, 5))
pos = list(range(len(Quarter)))  # position
bar_width = 0.15
opacity = 0.8
rects1 = plt.bar(pos, Quarter_return_pure, bar_width, alpha=opacity, color='#EE3224', label='Pure Market Trending')
rects2 = plt.bar([p + bar_width for p in pos], Quarter_return1, bar_width, alpha=opacity, color='#F78F1E', label='Market Trending + Weight Scheme 1')
rects3 = plt.bar([p + bar_width*2 for p in pos], Quarter_return2, bar_width, alpha=opacity, color='#FFC222', label='Market Trending + Weight Scheme 2')
rects4 = plt.bar([p + bar_width*3 for p in pos], Quarter_return_optimized, bar_width, alpha=opacity, color='#FBA293', label='Market Trending + Weight Optimized')
rects5 = plt.bar([p + bar_width*4 for p in pos], Quarter_return_benchmark, bar_width, alpha=opacity, color='#FBD693', label='Benchmark')
ax.set_xticks([p + 2 * bar_width for p in pos])  # Set the position of the x ticks
ax.set_xticklabels(Quarter)  # Set the labels for the x ticks
fig.autofmt_xdate()  # Rotate values to see more clearly
plt.xlabel('Semiannual Quarters')
plt.ylabel("%")
plt.title('Semiannual Return')
plt.xlim(min(pos) - bar_width, max(pos) + bar_width * 4)  # Setting the x-axis limits
plt.legend(loc='best')
plt.tight_layout()
plt.grid()
fig.savefig(directory + market + " Semiannual Return Combined" + ".png")
# plt.show()

# Change the year before graph
if 2000 < year_input < 2018:
    # timestamp
    timeindex1 = 0
    year1 = 0
    while timeindex1 < len(timestamp) and year1 < year_input:
        time1 = timestamp[timeindex1]
        year1 = int(time1.split("/")[2])
        timeindex1 += 1
    timestamp = timestamp[timeindex1 - 1:]
    price = price[timeindex1 - 1:]
    portfolio_pure = portfolio_pure[timeindex1 - 1:]
    portfolio_weight1 = portfolio_weight1[timeindex1 - 1:]
    portfolio_weight2 = portfolio_weight2[timeindex1 - 1:]
    portfolio_weight_optimized = portfolio_weight_optimized[timeindex1 - 1:]
    # Date Index
    timeindex2 = 0
    year2 = 0
    while timeindex2 < len(date_index) and year2 < year_input:
        time2 = date_index[timeindex2]
        year2 = int(time2.split("/")[2])
        timeindex2 += 1
    date_index = date_index[timeindex2 - 1:]
    price_index = price_index[timeindex2 - 1:]
    temp = []
    for i in range(len(price_index)):
        temp.append(price_index[i]/price_index[0])
    price_index = temp

# Find correct date to match dates for market index and our portfolio
correctdate = []
correctdate2 = []
for i in range(len(timestamp)):
    for j in range(len(date_index)):
        if date_index[j] == timestamp[i]:
            correctdate.append(i)
            correctdate2.append(j)

# Prepare for a graph
x_axis = []
y = []  # Pure Market Trending
strategy1 = []  # Weight Scheme 1
strategy2 = []  # Weight Scheme 2
strategy3 = []  # Weight Optimized
y_index = []  # Market Index Portfolio

# Normalize to initial amount and fill up values for y-axis by searching for correct dates
portfolio_pure = functions.relative(portfolio_pure)
portfolio_weight1 = functions.relative(portfolio_weight1)
portfolio_weight2 = functions.relative(portfolio_weight2)
portfolio_weight_optimized = functions.relative(portfolio_weight_optimized)
for i in range(len(correctdate)):
    correctindex = correctdate[i]
    y.append(portfolio_pure[correctindex])
    strategy1.append(portfolio_weight1[correctindex])
    strategy2.append(portfolio_weight2[correctindex])
    strategy3.append(portfolio_weight_optimized[correctindex])
for j in range(len(correctdate2)):
    correctindex = correctdate2[j]
    y_index.append(price_index[correctindex])
    x_axis.append(date_index[correctindex])

title_stats = ["Pure Market Trending", "Weight Scheme 1", "Weight Scheme 2", 'Weight Optimized', market]
stats = [y, strategy1, strategy2, strategy3, y_index]
for count in range(len(stats)):
    calculate = functions.percentagelist(stats[count])
    cumulative_return = functions.percentageChange(stats[count][0], stats[count][-1])
    annualized_return = average(calculate) * 252
    annuzlied_volatility = SD(calculate) * (252 ** (1 / 2))
    sharpe_ratio = sharpe(calculate)
    max_drawdown = functions.findMin(calculate)
    print(title_stats[count])
    print("Cumulative Return:", round(cumulative_return, 4), "%")
    print("Annualized Return:", round(annualized_return, 4), "%")
    print("Annualized Volatility:", round(annuzlied_volatility, 4), "%")
    print("Sharpe Ratio:", round(sharpe_ratio, 4))
    print("Max Drawdown:", round(max_drawdown, 4), "%")
    print("\n")
    stats_line = [title_stats[count], cumulative_return, annualized_return, annuzlied_volatility, sharpe_ratio, max_drawdown]
    stats_matrix.append(stats_line)

# Write csv file of stats matrix inside result folder
with open(directory + 'Stats from ' + str(year_input) + '.csv', "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in stats_matrix:
        writer.writerow(val)

# Plot a graph
y_index = functions.relative(y_index)  # Normalize benchmark
fig = plt.figure()  # Create a figure
ax = plt.axes()  # Create axis
if len(x_axis) == len(y):
    # bou = int(len(y)/5)
    plt.plot(x_axis, functions.relative(y), label='Pure Market Trending')
    plt.plot(x_axis, functions.relative(strategy1), label='Market Trending + Weight 1')
    plt.plot(x_axis, functions.relative(strategy2), label='Market Trending + Weight 2')
    plt.plot(x_axis, functions.relative(strategy3), label='Market Trending + Weight Optimized')
    plt.plot(x_axis, functions.relative(y_index), label=market)
    plt.grid(True)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))  # Set Maximum number of x-axis values to show
    fig.autofmt_xdate()  # Rotate values to see more clearly
    legend = plt.legend(loc='best')
    title = "Trend Following: Growth of " + str(initial_amount / 1000000) + " million"
    plt.title(title)
    plt.ylabel("Cumulative Return")
    fig.savefig(directory + market + " Market Timing from " + str(year_input) + ".png")
    # plt.show()

# Check When our strategy performs worse than benchmark
worse1 = []
worse2 = []
# period = 60  # 3 Months
period = 20  # 1 Month
for i in range(1, len(x_axis)):
    if i % period == 0:
        benchmark = (y_index[i] - y_index[i - period]) / y_index[i - period]
        y1 = (strategy1[i] - strategy1[i - period]) / strategy1[i - period]
        y2 = (strategy2[i] - strategy2[i - 1]) / strategy2[i - period]
        # Market Timing + Weight Scheme 1
        if benchmark > y1:
            worse1.append(i)
        # Market Timing + Weight Scheme 2
        if benchmark > y2:
            worse2.append(i)
#print("\n")
#print("When Market Timing + Weight Scheme 1 performs worse than benchmark:")
#for i in range(len(worse1)):
#    day = x_axis[worse1[i]].split('/')[0]
#    month = x_axis[worse1[i]].split('/')[1]
#    year = x_axis[worse1[i]].split('/')[2]
#    if int(day) < 15:
#        if int(month)-1 != 0:
#            print(year, int(month)-1)
#        else:
#            print(int(year)-1, 12)
#    else:
#        print(year, month)
#print("\n")
#print("When Market Timing + Weight Scheme 2 performs worse than benchmark:")
#for i in range(len(worse2)):
#    day = x_axis[worse2[i]].split('/')[0]
#    month = x_axis[worse2[i]].split('/')[1]
#    year = x_axis[worse2[i]].split('/')[2]
#    if int(day) < 15:
#        if int(month)-1 != 0:
#            print(year, int(month)-1)
#        else:
#            print(int(year)-1, 12)
#    else:
#        print(year, month)

print("Market Timing + Weight Scheme 1: ", round(len(worse1)/(len(x_axis) / period) * 100, 4), "%: performed worse than", market)
print("Market Timing + Weight Scheme 2: ", round(len(worse2)/(len(x_axis) / period) * 100, 4), "%: performed worse than", market)

# Add Vertical Shadows
if len(x_axis) == len(y):
    fig = plt.figure(figsize=(20, 10))
    # When Market Timing + Weight Scheme 1 performs worse than benchmark
    ax = plt.subplot(1, 2, 1)
    ax.plot(x_axis, functions.relative(y_index), label=market)
    ax.plot(x_axis, functions.relative(strategy1), label='Market Trending + Weight 1')
    ax.grid(True)
    ax.legend(loc='best')
    for xc in worse1:
        ax.axvline(x=xc, alpha=0.1)
        ax.axvspan(xc - period, xc, alpha=0.1)
    ax.xaxis.set_major_locator(plt.MaxNLocator(2))  # Set Maximum number of x-axis values to show
    # When Market Timing + Weight Scheme 2 performs worse than benchmark
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(x_axis, functions.relative(y_index), label=market)
    ax2.plot(x_axis, functions.relative(strategy2), label='Market Trending + Weight 2')
    ax2.grid(True)
    ax2.legend(loc='best')
    for xc in worse2:
        ax2.axvline(x=xc, alpha=0.1)
        ax2.axvspan(xc - period, xc, alpha=0.1)
    ax2.xaxis.set_major_locator(plt.MaxNLocator(2))  # Set Maximum number of x-axis values to show
    fig.savefig(directory + market + " Performance from " + str(year_input) + ".png")
