# This is to calculate cumulative returns and plot graphs
import functions
import csv
import numpy as np
import os
import time
import matplotlib.pyplot as plt
#import random_scores

# Initialization
market = "HSCI"
oneyear = 252  # 1 year = 252 trading days
timestamp = []
date_index = []
price_index = []
store_index = []  # when quarter changes
list_cumulative = []
list_sharpe = []
weight_name = []
# Variables
initial_amount = 1000000  # 1M
year_input = 2006
# Functions
replace = functions.replace_na
average = functions.findaverage
SD = functions.findSD
maxconsecutive = functions.MaxConsecutive
winrate = functions.WinRate
sharpe = functions.Sharpe

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
    ticker = price[0][1:]
    price = replace(price)
    price = price[1:]
    for i in range(len(price)):
        timestamp.append(price[i][0])

# Delete Prices before 2006
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


def optimization(weight_scheme):
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
    cash_amount = 0
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
        for stock in range(len(ticker)):
            gain = functions.percentageChange(float(price[index - 1][stock + 1]), float(price[index][stock + 1]))
            if answer == 0:
                stockweight = 0
            else:
                w = float(weight_scheme[answer][stock + 1])
                stockweight = w / 100
            if bool_quarter:  # When quarter changes
                if index == 0:
                    value = initial_amount * stockweight
                else:
                    value = column_return[index - 1] * stockweight
                    cash_hand -= w
            else:  # When quarter does not change
                value = float(resultMatrix[index][stock + 1]) * (1 + gain / 100)
            resultMatrix[index + 1][stock + 1] = value  # Fill in the money allocation instead of percentage change
            aggregate += value
            # Calculate number of stocks to be invested
            if stockweight != 0 and float(price[index][stock + 1]) != 0:
                stockcount += 1
        # Calculate cash in hand when remaining weight is not zero
        if bool_quarter and cash_hand < 100:
            cash_amount = cash_hand * column_return[index - 1] / 100
        if bool_quarter and cash_hand == 100:
            cash_amount = 0
        if aggregate == 0:
            aggregate = column_return[index - 1]
            if column_return[index - 1] == 0 or index == 0:
                aggregate = initial_amount
        aggregate += cash_amount
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
    list_cumulative.append(cumulative_return)
    # Calculate related stats
    sharpe_ratio = sharpe(daily_return)
    list_sharpe.append(sharpe_ratio)
    print(numberofday, "days;", round(cumulative_return, 4), "%, Sharpe = ", round(sharpe_ratio, 4))

    return column_return


directory = 'Random Weights/'
count = 0
# Loop through files in the directory
for filename in os.listdir(directory):
    start = time.time()
    weight_name.append(filename)
    print(filename[:-4])  # delete '.csv'
    # Import prices of each stock
    with open(directory + filename) as f:
        reader = csv.reader(f)
        weight = [rows for rows in reader]
    optimization(weight)
    end = time.time()
    print("Trial", count + 1, ":", round(end - start, 4), "seconds")
    print("\n")
    count += 1

print(list_cumulative)
print(list_sharpe)
# Cumulative Return
max_index = functions.findMaxIndex(list_cumulative)
min_index = functions.findMinIndex(list_cumulative)
max_name = weight_name[max_index]
min_name = weight_name[min_index]
# Sharpe Ratio
max_index_sharpe = functions.findMaxIndex(list_sharpe)
min_index_sharpe = functions.findMinIndex(list_sharpe)
max_name_sharpe = weight_name[max_index_sharpe]
min_name_sharpe = weight_name[min_index_sharpe]

print('Maximum Cumulative Return:', round(max(list_cumulative), 4), "%", " - ", max_name)
print("Minimum Cumulative Return:", round(min(list_cumulative), 4), "%", " - ", min_name)
print('Maximum Sharpe Ratio:', round(max(list_sharpe), 4), " - ", max_name_sharpe)
print("Minimum Sharpe Ratio:", round(min(list_sharpe), 4), " - ", min_name_sharpe)
print("Original:", round(list_cumulative[-1], 4), "%; Annual Sharpe = ", round(list_sharpe[-1], 4))
print("\n")

# Import Max Cumulative Return Weight
with open(directory + max_name) as f:
    reader = csv.reader(f)
    max_weight = [rows for rows in reader]
# Write Optimized Weight CSV inside Stock Picking Result
with open('Stock Picking Result/' + 'weight optimized' + '.csv', 'w') as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in max_weight:
        writer.writerow(val)

# Import Min Cumulative Return Weight
with open(directory + min_name) as f:
    reader = csv.reader(f)
    min_weight = [rows for rows in reader]

# Import Max Sharpe Ratio Weight
with open(directory + max_name_sharpe) as f:
    reader = csv.reader(f)
    max_weight_sharpe = [rows for rows in reader]

# Import Min Sharpe Ratio Weight
with open(directory + min_name_sharpe) as f:
    reader = csv.reader(f)
    min_weight_sharpe = [rows for rows in reader]

# Import Original Weight
with open(directory + weight_name[-1]) as f:
    reader = csv.reader(f)
    original_weight = [rows for rows in reader]

print("Maximum Cumulative Return")
max_cumulative = optimization(max_weight)
print("Minimum Cumulative Return")
min_cumulative = optimization(min_weight)
print("Maximum Sharpe Ratio")
max_sharpe = optimization(max_weight_sharpe)
print("Minimum Sharpe Ratio")
min_sharpe = optimization(min_weight_sharpe)
print("Original")
original = optimization(original_weight)

# New directory
directory_graph = 'Optimization_Piotroski/'
if not os.path.exists(directory_graph):
    os.makedirs(directory_graph)

# Prepare for a graph
x_axis = []
y_index = []
y = []  # original
# Cumulative Return
y_max = []
y_min = []
# Sharpe Ratio
y_max_sharpe = []
y_min_sharpe = []

# Match Timestamp
correctdate = []
correctdate2 = []
for i in range(len(timestamp)):
    for j in range(len(date_index)):
        if date_index[j] == timestamp[i]:
            correctdate.append(i)
            correctdate2.append(j)
            x_axis.append(timestamp[i])
            y.append(original[i])
            y_max.append(max_cumulative[i])
            y_min.append(min_cumulative[i])
            y_max_sharpe.append(max_sharpe[i])
            y_min_sharpe.append(min_sharpe[i])
            y_index.append(price_index[j])

# Draw Graphs
fig = plt.figure(figsize=(15, 6))  # Create a figure
title = "Piotroski Optimization: Growth of " + str(initial_amount / 1000000) + " million"
plt.suptitle(title, fontsize=18)
# Cumulative Return
ax1 = plt.subplot(1, 2, 1)  # Create axis
ax1.plot(x_axis, functions.relative(y_max), label=max_name[:-4] + ': Max Cumulative Return')
ax1.plot(x_axis, functions.relative(y_min), label=min_name[:-4] + ': Min Cumulative Return')
ax1.plot(x_axis, functions.relative(y), label='original weight')
ax1.plot(x_axis, functions.relative(y_index), label=market)
ax1.xaxis.set_major_locator(plt.MaxNLocator(5))  # Set Maximum number of x-axis values to show
ax1.legend(loc='best')
ax1.grid(True)
plt.title("Max Cumulative Return", fontsize=12)
plt.ylabel("Cumulative Return", fontsize=8)
# Sharpe Ratio
ax2 = plt.subplot(1, 2, 2)  # Create axis
ax2.plot(x_axis, functions.relative(y_max_sharpe), label=max_name_sharpe[:-4] + ': Max Sharpe Ratio')
ax2.plot(x_axis, functions.relative(y_min_sharpe), label=min_name_sharpe[:-4] + ': Min Sharpe Ratio')
ax2.plot(x_axis, functions.relative(y), label='original weight')
ax2.plot(x_axis, functions.relative(y_index), label=market)
ax2.xaxis.set_major_locator(plt.MaxNLocator(5))  # Set Maximum number of x-axis values to show
ax2.legend(loc='best')
ax2.grid(True)
plt.title("Max Sharpe Ratio", fontsize=12)
plt.ylabel("Cumulative Return", fontsize=8)
# Adjust positions
plt.tight_layout()
plt.subplots_adjust(wspace=0.15, top=0.9)
# Save figure
fig.savefig(directory_graph + market + " Piotroski Optimization.png")
plt.show()
