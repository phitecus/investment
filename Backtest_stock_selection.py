import csv
import functions
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import weight1
import weight2

weight_1 = weight1.weightMatrix
weight_2 = weight2.weightMatrix

# Variables
initial_amount = 1000000  # 1M
year_input = 2008

# Initialization
market = "HSCI"
oneyear = 252  # 1 year = 252 trading days
timestamp = []
date_index = []
price_index = []
store_index = []  # when quarter changes
stats_matrix = [[0 for x in range(6)] for y in range(1)]
# Functions
replace = functions.replace_na
average = functions.findaverage
SD = functions.findSD
maxconsecutive = functions.MaxConsecutive
winrate = functions.WinRate
sharpe = functions.Sharpe
head_stats = [str(year_input), "Cumulative Return (%)", "Annualized Return (%)", "Annualized Volatility (%)", "Sharpe Ratio", "Max Drawdown (%)"]
for i in range(len(stats_matrix[0])):
    stats_matrix[0][i] = head_stats[i]

# Set the directory
directory = "Stock Picking Result/"
if not os.path.exists(directory):
    os.makedirs(directory)

# Import market index
with open(market+'.csv') as f:
    reader = csv.reader(f)
    index = [rows for rows in reader]
    for i in range(len(index)):
        date_index.append(index[i][0])
        price_index.append(float(index[i][1]))

# Import prices of each stock
with open('prices.csv') as g:
    reader = csv.reader(g)
    price = [rows for rows in reader]
    price = replace(price)
    price = price[1:]
    for i in range(len(price)):
        timestamp.append(price[i][0])

# Find the timestamp of designated year and delete previous ones
if year_input <= 2000 or year_input >= 2018:
    print("Use Default Year")
    year_input = 2000
else:
    # timestamp
    timeindex1 = 0
    year1 = 0
    while timeindex1 < len(timestamp) and year1 < year_input:
        time1 = timestamp[timeindex1]
        year1 = int(time1.split("/")[2])
        timeindex1 += 1
    timestamp = timestamp[timeindex1 - 1:]
    price = price[timeindex1 - 1:]
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

print("\n")
print(market, "BACK TEST", "start from", year_input)
ticker = weight1.ticker


def backtest(weight_scheme, number):
    numberofday = 0
    quarter_list = []

    # Initialize matrix full of zeros
    resultMatrix = [[0 for x in range(len(ticker)+5)] for y in range(len(timestamp)+1)]

    # Change the first row of result matrix to show headers
    firstrow = resultMatrix[0]
    secondrow = resultMatrix[1]
    for i0 in range(len(firstrow)):
        if i0 == 0:
            firstrow[i0] = ""
        elif i0 < len(ticker)+1:
            firstrow[i0] = ticker[i0-1]
        elif i0 == len(ticker)+1:
            firstrow[i0] = "Daily Return"
        elif i0 == len(ticker)+2:
            firstrow[i0] = "Cumulative"
        elif i0 == len(ticker)+3:
            firstrow[i0] = "Stock Count"
        elif i0 == len(ticker)+4:
            firstrow[i0] = "Portfolio"
            secondrow[i0] = initial_amount

    # Change the first column of result matrix to show time index
    for row in range(len(resultMatrix)):
        if row == 0:
            resultMatrix[row][0] = ""
        else:
            resultMatrix[row][0] = timestamp[row-1]

    resultMatrix = np.array(resultMatrix)
    column_return = np.array(resultMatrix[:, -1][1:])  # Portfolio
    column_return = np.asfarray(column_return, float)
    column_return[0] = initial_amount

    # Loop through time index
    for index in range(len(timestamp)):
        bool_quarter = False
        stockcount = 0
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
        for stock in range(len(ticker)):
            gain = functions.percentageChange(float(price[index - 1][stock + 1]), float(price[index][stock + 1]))
            if answer == 0:
                stockweight = 0
            else:
                stockweight = float(weight_scheme[answer][stock + 1])/100
            if bool_quarter:  # When quarter changes
                if index == 0:
                    value = initial_amount * stockweight
                else:
                    value = column_return[index-1] * stockweight
            else:  # When quarter does not change
                value = float(resultMatrix[index][stock+1]) * (1 + gain / 100)
            resultMatrix[index + 1][stock + 1] = value  # Fill in the money allocation instead of percentage change
            aggregate += value
            # Calculate number of stocks to be invested
            if stockweight != 0 and float(price[index][stock + 1]) != 0:
                stockcount += 1
        if aggregate == 0:
            aggregate = column_return[index-1]
            if column_return[index-1] == 0 or index == 0:
                aggregate = initial_amount
        column_return[index] = aggregate

        # Stock Count
        resultMatrix[index + 1][len(ticker) + 3] = stockcount
        numberofday += 1

    resultMatrix[:, -1][1:] = column_return
    daily_return = functions.percentagelist(column_return)
    resultMatrix[:, -4][2:] = daily_return
    cumulative = functions.cumulative_list(daily_return)
    resultMatrix[:, -3][2:] = cumulative
    portfolio_list = column_return

    # Take the last index to get cumulative_return
    cumulative_return = cumulative[-1]  # Final cumulative return

    # Calculate related stats
    head = "Weight Scheme " + str(number)
    print(head, ":")
    print(numberofday, "days;", round(cumulative_return, 4), "%")
    print("\n")
    print("Average Daily Return =", round(average(daily_return), 4), "%")
    print("SD of Daily Return =", round(SD(daily_return), 4), "%")
    annual_return = average(daily_return) * oneyear
    print("Annualized Return =", round(annual_return, 4), "%")
    annual_volatility = SD(daily_return) * (oneyear ** (1 / 2))
    print("Annualized Volatility =", round(annual_volatility, 4), "%")
    sharpe_ratio = sharpe(daily_return)
    print("Annual Sharpe Ratio: ", round(sharpe_ratio, 4))
    print("Max Consecutive Loss = ", maxconsecutive(daily_return), "days")
    print("Win Rate:", round(winrate(daily_return), 4), "%")
    max_drawdown = functions.findMin(daily_return)
    print("Max Drawdown:", round(max_drawdown, 4), "%")
    print("\n")
    stats_line = [head, cumulative_return, annual_return, annual_volatility, sharpe_ratio, max_drawdown]
    stats_matrix.append(stats_line)

    if number == 2:
        # Calculate stats for market index
        daily_return_index = functions.percentagelist(price_index)
        cumulative_index = functions.percentageChange(price_index[0], price_index[-1])
        print(market, "Index: ")
        print("Cumulative =", round(cumulative_index, 4), "%")
        print("Average Daily Return =", round(average(daily_return_index), 4), "%")
        print("SD of Daily Return =", round(SD(daily_return_index), 4), "%")
        annual_return = average(daily_return_index) * oneyear
        print("Annualized Return =", round(annual_return, 4), "%")
        annual_volatility = SD(daily_return_index) * (oneyear ** (1 / 2))
        print("Annualized Volatility =", round(annual_volatility, 4), "%")
        sharpe_ratio = sharpe(daily_return_index)
        print("Annual Sharpe Ratio: ", round(sharpe_ratio, 4))
        print("Max Consecutive Loss = ", maxconsecutive(daily_return_index), "days")
        print("Win Rate:", round(winrate(daily_return_index), 4), "%")
        max_drawdown = functions.findMin(daily_return)
        print("Max Drawdown:", round(max_drawdown, 4), "%")
        stats_line = [market, cumulative_index, annual_return, annual_volatility, sharpe_ratio, max_drawdown]
        stats_matrix.append(stats_line)

    # Find correct date to match dates for market index and our portfolio
    correctdate = []
    correctdate2 = []
    for i in range(len(timestamp)):
        for j in range(len(date_index)):
            if date_index[j] == timestamp[i]:
                correctdate.append(i)
                correctdate2.append(j)

    # Prepare for a graph
    x_axis = date_index
    y = []  # Strategy Portfolio
    y_index = []  # Market Index Portfolio

    # Normalize to initial amount and fill up values for y-axis by searching for correct dates
    relative_portfolio = functions.relative(portfolio_list)
    for i in range(len(correctdate)):
        correctindex = correctdate[i] - 1
        y.append(relative_portfolio[correctindex])
    for j in range(len(correctdate2)):
        correctindex = correctdate2[j]
        y_index.append(price_index[correctindex])
    y_index = functions.relative(y_index)

    # Plot a graph
    fig = plt.figure()  # Create a figure
    ax = plt.axes()  # Create axis
    if len(x_axis) == len(y):
        plt.plot(x_axis, y, label='Weight Scheme '+str(number))
        plt.plot(x_axis, y_index, label=market)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))  # Set Maximum number of x-axis values to show
        fig.autofmt_xdate()  # Rotate values to see more clearly
        title = "Stock Selection: Growth of " + str(initial_amount / 1000000) + " million"
        plt.title(title)
        plt.ylabel("Cumulative Return")
        fig.savefig(directory + market + " Stock Selection Weight Scheme " + str(number) + ".png")
        plt.show()

    # Write csv file of result matrix inside result folder
    title = directory + 'weight scheme ' + str(number)
    with open(title + ' result.csv', "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in resultMatrix:
            writer.writerow(val)


backtest(weight_1, 1)
backtest(weight_2, 2)

# Write csv file of stats matrix inside result folder
with open(directory + 'Stats from ' + str(year_input) + '.csv', "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in stats_matrix:
        writer.writerow(val)

df = pd.read_csv((directory + "weight scheme 1 result.csv"), header=None, low_memory=False)
header = df.values[0][1:]  # First Row (skip first column)
result = np.array(df[1:])  # Skip First Row
timestamp = result[:, 0]  # First Column
portfolio = result[:, len(header)]  # Last Column
portfolio = portfolio[1:]  # Delete 'Start From Below' part
timestamp = timestamp[1:]

df2 = pd.read_csv((directory + "weight scheme 2 result.csv"), header=None, low_memory=False)
result2 = np.array(df2[1:])  # Skip First Row
portfolio2 = result2[:, len(header)]  # Last Column
portfolio2 = portfolio2[1:]  # Delete 'Start From Below' part

# Calculate Quarterly Return (HK: semiannually)
quarter_matrix = [[0 for x in range(len(store_index))] for y in range(3)]
quarter_matrix[0][0] = "Semiannual Return"
quarter_matrix[1][0] = "Weight Scheme 1"
quarter_matrix[2][0] = "Weight Scheme 2"
for i in range(len(store_index) - 1):
    quarter_matrix[0][i + 1] = store_index[i][1]
    quarter_matrix[1][i + 1] = functions.percentageChange(float(portfolio[store_index[i][0]]),
                                                          float(portfolio[store_index[i + 1][0]]))
    quarter_matrix[2][i + 1] = functions.percentageChange(float(portfolio2[store_index[i][0]]),
                                                          float(portfolio2[store_index[i + 1][0]]))

Quarter = quarter_matrix[0][1:]
Quarter_return1 = quarter_matrix[1][1:]
Quarter_return2 = quarter_matrix[2][1:]

# Draw quarterly return bar graphs
fig = plt.figure()  # Create a figure
ax = plt.axes()  # Create axis
plt.bar(Quarter, Quarter_return1, align='center', alpha=0.5)  # Draw Bar graph
ax.xaxis.set_major_locator(plt.MaxNLocator(8))  # Set Maximum number of x-axis values to show
fig.autofmt_xdate()  # Rotate values to see more clearly
plt.ylabel('%')
plt.title('Weight Scheme 1')
fig.savefig(directory + market + " Semiannual Return Weight 1" + ".png")
plt.show()
# Weight 2
fig = plt.figure()  # Create a figure
ax = plt.axes()  # Create axis
plt.bar(Quarter, Quarter_return2, align='center', alpha=0.5)  # Draw Bar graph
ax.xaxis.set_major_locator(plt.MaxNLocator(8))  # Set Maximum number of x-axis values to show
fig.autofmt_xdate()  # Rotate values to see more clearly
plt.ylabel('%')
plt.title('Weight Scheme 2')
fig.savefig(directory + market + " Semiannual Return Weight 2" + ".png")
plt.show()

# Compare bar charts
fig, ax = plt.subplots(figsize=(10, 5))
pos = list(range(len(Quarter)))  # position
bar_width = 0.35
opacity = 0.8
rects1 = plt.bar(pos, Quarter_return1, bar_width, alpha=opacity, label='Weight Scheme 1')
rects2 = plt.bar([p + bar_width for p in pos], Quarter_return2, bar_width, alpha=opacity, label='Weight Scheme 2')
ax.xaxis.set_major_locator(plt.MaxNLocator(5))  # Set Maximum number of x-axis values to show
ax.set_xticks([p + bar_width for p in pos])  # Set the position of the x ticks
ax.set_xticklabels(Quarter)  # Set the labels for the x ticks
fig.autofmt_xdate()  # Rotate values to see more clearly
plt.xlabel('Semiannual Quarters')
plt.ylabel("%")
plt.title('Semiannual Return')
plt.legend(loc='best')
plt.tight_layout()
plt.grid()
fig.savefig(directory + market + " Semiannual Return Combined" + ".png")
plt.show()

# Write csv file of result matrix inside result folder
with open(directory + 'Semiannual Return.csv', "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in quarter_matrix:
        writer.writerow(val)

# Find correct date to match dates for market index and our portfolio
correctdate = []
correctdate2 = []
for i in range(len(timestamp)):
    for j in range(len(date_index)):
        if date_index[j] == timestamp[i]:
            correctdate.append(i)
            correctdate2.append(j)

# Prepare for a graph
x_axis = date_index
y1 = []  # Weight Scheme1 Portfolio
y2 = []  # Weight Scheme2 Portfolio
y_index = []  # Market Index Portfolio

# Normalize to initial amount and fill up values for y-axis by searching for correct dates
relative_portfolio = functions.relative(portfolio)
relative_portfolio2 = functions.relative(portfolio2)
for i in range(len(correctdate)):
    correctindex = correctdate[i]
    y1.append(relative_portfolio[correctindex])
    y2.append(relative_portfolio2[correctindex])

for j in range(len(correctdate2)):
    correctindex = correctdate2[j]
    y_index.append(price_index[correctindex])
y_index = functions.relative(y_index)

# Plot a graph
fig = plt.figure()  # Create a figure
ax = plt.axes()  # Create axis
plt.plot(x_axis, y1, label='Weight Scheme 1')
plt.plot(x_axis, y2, label='Weight Scheme 2')
plt.plot(x_axis, y_index, label=market)
ax.xaxis.set_major_locator(plt.MaxNLocator(5))  # Set Maximum number of x-axis values to show
fig.autofmt_xdate()  # Rotate values to see more clearly
legend = plt.legend(loc='best')
title = "Stock Selection: Growth of " + str(initial_amount / 1000000) + " million"
plt.title(title)
plt.ylabel("Cumulative Return")
fig.savefig(directory + market + " Stock Selection Combined.png")
plt.show()

