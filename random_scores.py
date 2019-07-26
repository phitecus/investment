# Generate Random Variables, Random Scores, Random Scores after sector selection, Weights according to the scores
import functions
import numpy as np
import pandas as pd
import os
from random import randint
import csv
import time
import statistics

# Import prices of each stock
with open('prices.csv') as f:
    reader = csv.reader(f)
    price = [rows for rows in reader]
    price = functions.replace_na(price)
    ticker = price[0][1:]
    price = price[1:]
price = np.array(price)

iteration = 427
header = ["Tickers", "ROE>5", "Increasing Revenue", "Positive OCF", "Increasing EPS", 'Quality of Earnings', "Long Term Debt vs Asset", "Increasing Net Income", "Market Cap > Median", "Increasing Asset Turnover"]
# Import relevant fundamental indicators
market_cap = []
asset_turnover = []
EPS = []
LT_Debt = []
net_income = []
operating_cf = []
revenue = []
ROE = []
files = ["market cap", "Asset Turnover Ratio", "EPS", "Long Term Debt Ratio", "Net Income", "operating cash flow",
         "revenue", "ROE"]
for file in range(len(files)):
    with open(files[file] + '.csv') as g:
        reader = csv.reader(g)
        if file == 0:
            market_cap = [rows for rows in reader]
            sector = market_cap[1][1:]
            market_cap = functions.Check_number(market_cap[2:])  # Delete headers and sectors
        elif file == 1:
            asset_turnover = [rows for rows in reader]
            asset_turnover = functions.Check_number(asset_turnover[1:])
        elif file == 2:
            EPS = [rows for rows in reader]
            EPS = functions.Check_number(EPS[1:])
        elif file == 3:
            LT_Debt = [rows for rows in reader]
            LT_Debt = functions.Check_number(LT_Debt[1:])
        elif file == 4:
            net_income = [rows for rows in reader]
            net_income = functions.Check_number(net_income[1:])
        elif file == 5:
            operating_cf = [rows for rows in reader]
            operating_cf = functions.Check_number(operating_cf[1:])
        elif file == 6:
            revenue = [rows for rows in reader]
            revenue = functions.Check_number(revenue[1:])
        else:
            ROE = [rows for rows in reader]
            ROE = functions.Check_number(ROE[1:])

# Select a stock pool based on previous sector screening
directory = "Sector Selection Result/"
df = pd.read_csv((directory + "Ranking [1 and 0].csv"), header=None)
# header = df.values[0][1:]
ranking = np.array(df[1:])
date_ranking = ranking[:, 0]  # First Column

# Set the directory
directory_score = "Random Scores/"
if not os.path.exists(directory_score):
    os.makedirs(directory_score)


def random_scores():
    # Randomly generate parameters
    equalweight = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    list_variables = [equalweight]
    boundary = 1000
    for stock in range(len(ticker)):
        add = []
        # 1. Higher EPS than previous period
        var1 = randint(0, boundary)
        add.append(var1)
        # 2. Return of Equity in the current period > 5
        var2 = randint(0, boundary)
        add.append(var2)
        # 3. Higher revenue than previous period
        var3 = randint(0, boundary)
        add.append(var3)
        # 4. Higher net income than previous period
        var4 = randint(0, boundary)
        add.append(var4)
        # 5. Lower Ratio of Long term Debt than previous period
        var5 = randint(0, boundary)
        add.append(var5)
        # 6. Market Cap > Median
        var6 = randint(0, boundary)
        add.append(var6)
        # 7. Higher Asset Turnover Ratio than previous period
        var7 = randint(0, boundary)
        add.append(var7)
        # 8. Positive Operating Cash Flow
        var8 = randint(0, boundary)
        add.append(var8)
        # 9. Quality of Earnings (Operating Cash Flow > Net Income)
        var9 = randint(0, boundary)
        add.append(var9)
        # Divide by total number
        total = sum(add)
        add2 = [9 * i / total for i in add]
        list_variables.append(add2)

    # New directory
    directory_variable = 'Optimization_Piotroski/'
    if not os.path.exists(directory_variable):
        os.makedirs(directory_variable)
    # Count number of files in a folder
    matrix_files = os.listdir(directory_variable)
    number_files = len(matrix_files)

    matrix_variable = [[1 for x in range(len(header) + 1)] for y in range(len(ticker) + 1)]
    matrix_variable[0] = header  # Fill the first row as the header
    if number_files != 0:
        for i in range(1, len(matrix_variable)):
            row = matrix_variable[i]
            row[0] = ticker[i-1]
            row[1:] = list_variables[i]
        # Write Variable Matrix CSV
        title = 'Variables ' + str(number_files)
        with open(directory_variable + title + '.csv', 'w') as output:
            writer = csv.writer(output, lineterminator='\n')
            for val in matrix_variable:
                writer.writerow(val)
    else:
        for i in range(1, len(matrix_variable)):
            row = matrix_variable[i]
            row[0] = ticker[i - 1]
            title = 'Equal Weight Variables'
            with open(directory_variable + title + '.csv', 'w') as output:
                writer = csv.writer(output, lineterminator='\n')
                for val in matrix_variable:
                    writer.writerow(val)

    # Evaluate score according to criteria: initialize score matrix
    scoreMatrix = [[0 for x in range(len(ticker) + 1)] for y in range(len(date_ranking) + 1)]

    # Change the first row of result matrix to show headers
    firstrow = scoreMatrix[0]
    for i in range(len(firstrow)):
        if i == 0:
            firstrow[i] = ""
        elif i < len(ticker) + 1:
            firstrow[i] = ticker[i - 1]
    # Change the first column of result matrix to show time index
    for i in range(len(scoreMatrix)):
        if i == 0:
            scoreMatrix[i][0] = ""
        else:
            scoreMatrix[i][0] = date_ranking[i - 1]

    # Count number of files in a folder
    matrix_files = os.listdir(directory_score)
    number_files = len(matrix_files)

    # Piotroski F Score
    for row in range(len(scoreMatrix)):
        for column in range(len(scoreMatrix[row])):
            correctdate = 0  # default
            correctdate2 = 0  # default
            # if number of files == 0, create equal weight score matrix
            if number_files == 0:
                correct_column = 0
            else:
                correct_column = column  # start from 1

            if row != 0 and column != 0:
                today = scoreMatrix[row][0]
                # 1. Higher EPS than previous period
                for index_a in range(len(EPS)):
                    if EPS[index_a][0] == today:
                        correctdate = index_a
                if EPS[correctdate][column] > EPS[correctdate - 1][column]:
                    scoreMatrix[row][column] += (list_variables[correct_column][0])
                correctdate = 0

                # 2. Return of Equity in the current period > 5
                for index_b in range(len(ROE)):
                    if ROE[index_b][0] == today:
                        correctdate = index_b
                if correctdate != 0:
                    if ROE[correctdate][column] > 5:
                        scoreMatrix[row][column] += (list_variables[correct_column][1])
                correctdate = 0

                # 3. Higher revenue than previous period
                for index_c in range(len(revenue)):
                    if revenue[index_c][0] == today:
                        correctdate = index_c
                if correctdate != 0:
                    if revenue[correctdate][column] > revenue[correctdate - 1][column]:
                        scoreMatrix[row][column] += (list_variables[correct_column][2])
                correctdate = 0

                # 4. Higher net income than previous period
                for index_d in range(len(net_income)):
                    if net_income[index_d][0] == today:
                        correctdate = index_d
                if correctdate != 0:
                    if net_income[correctdate][column] > net_income[correctdate - 1][column]:
                        scoreMatrix[row][column] += (list_variables[correct_column][3])
                correctdate = 0

                # 5. Lower Ratio of Long term Debt than previous period
                for index_e in range(len(LT_Debt)):
                    if LT_Debt[index_e][0] == today:
                        correctdate = index_e
                if correctdate != 0:
                    if LT_Debt[correctdate][column] < LT_Debt[correctdate - 1][column]:
                        scoreMatrix[row][column] += (list_variables[correct_column][4])
                correctdate = 0

                # 6. Market Cap > Median
                for index_f in range(len(market_cap)):
                    if market_cap[index_f][0] == today:
                        correctdate = index_f
                if correctdate != 0:
                    median = statistics.median(functions.delete_zero(market_cap[correctdate][1:]))
                    if market_cap[correctdate][column] > median:
                        scoreMatrix[row][column] += (list_variables[correct_column][5])
                correctdate = 0

                # 7. Higher Asset Turnover Ratio than previous period
                for index_g in range(len(asset_turnover)):
                    if asset_turnover[index_g][0] == today:
                        correctdate = index_g
                if correctdate != 0:
                    if asset_turnover[correctdate][column] > asset_turnover[correctdate - 1][column]:
                        scoreMatrix[row][column] += (list_variables[correct_column][6])
                correctdate = 0

                # 8. Positive Operating Cash Flow
                for index_h in range(len(operating_cf)):
                    if operating_cf[index_h][0] == today:
                        correctdate = index_h
                if correctdate != 0:
                    if operating_cf[correctdate][column] > 0:
                        scoreMatrix[row][column] += (list_variables[correct_column][7])

                # 9. Quality of Earnings (Operating Cash Flow > Net Income)
                for index_i in range(len(net_income)):
                    if net_income[index_i][0] == today:
                        correctdate2 = index_i
                if correctdate != 0 and correctdate2 != 0:
                    if operating_cf[correctdate][column] > net_income[correctdate2][column]:
                        scoreMatrix[row][column] += (list_variables[correct_column][8])

    # Write CSV files
    if number_files == 0:
        title = 'score euqal weight'
    else:
        title = 'score ' + str(number_files)
    with open(directory_score + title + '.csv', 'w') as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in scoreMatrix:
            writer.writerow(val)

    # Combine with Sector Selection
    for quarter in range(len(date_ranking)):
        for stock in range(len(ticker)):
            signal = 0
            if sector[stock] == "Consumer Discretionary":
                signal = int(ranking[quarter][1])
            if sector[stock] == "Consumer Staples":
                signal = int(ranking[quarter][2])
            if sector[stock] == "Energy":
                signal = int(ranking[quarter][3])
            if sector[stock] == "Financials":
                signal = int(ranking[quarter][4])
            if sector[stock] == "Health Care":
                signal = int(ranking[quarter][5])
            if sector[stock] == "Industrials":
                signal = int(ranking[quarter][6])
            if sector[stock] == "Information Technology":
                signal = int(ranking[quarter][7])
            if sector[stock] == "Materials":
                signal = int(ranking[quarter][8])
            if sector[stock] == "Real Estate":
                signal = int(ranking[quarter][9])
            if sector[stock] == "Telecommunication Services":
                signal = int(ranking[quarter][10])
            if sector[stock] == "Utilities":
                signal = int(ranking[quarter][11])
            scoreMatrix[quarter + 1][stock + 1] *= signal

    # New directory
    directory_score_after_sector = "Random Scores after Sector Selection/"
    if not os.path.exists(directory_score_after_sector):
        os.makedirs(directory_score_after_sector)
    # Count number of files in a folder
    matrix_files = os.listdir(directory_score_after_sector)
    number_files = len(matrix_files)

    if number_files == 0:
        title = 'score euqal weight'
    else:
        title = 'score ' + str(number_files)

    # Write csv file of result matrix after sector selection inside the folder
    with open(directory_score_after_sector + title + '.csv', 'w') as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in scoreMatrix:
            writer.writerow(val)

    # Generate Weight
    # Give at most 60% weight to the 9- score stocks, but each cannot take up more than 4%
    # Give remainder weight to the 8- score stocks, but each cannot take up more than 3%
    # Give remainder weight to the 7- score stocks, but each cannot take up more than 2.5%
    # Give remainder weight to the 6- score stocks, but each cannot take up more than (weight7)%
    for quarter in range(len(date_ranking)):
        number9 = 0
        number8 = 0
        number7 = 0
        number6 = 0
        weight9 = 0
        weight8 = 0
        weight7 = 0
        weight6 = 0
        remainder = 100
        row = scoreMatrix[quarter + 1][1:]
        for stock in range(len(ticker)):
            # Count Numbers
            if int(round(row[stock])) == 9:
                number9 += 1
            if int(round(row[stock])) == 8:
                number8 += 1
            if int(round(row[stock])) == 7:
                number7 += 1
            if int(round(row[stock])) == 6:
                number6 += 1
        # print(number9, number8, number7, number6)
        # Give at most 60% weight to the 9- score stocks, but each cannot take up more than 4%
        if number9 != 0:
            weight9 = min(4, 60 / number9)
            remainder -= (weight9 * number9)
        # Give remainder weight to the 8- score stocks, but each cannot take up more than 3%
        if number8 != 0:
            weight8 = min(3, remainder / number8)
            remainder -= (weight8 * number8)
        # Give remainder weight to the 7- score stocks, but each cannot take up more than 2.5%
        if number7 != 0:
            weight7 = min(2.5, remainder / number7)
            remainder -= (weight7 * number7)
        # Give remainder weight to the 6- score stocks
        if number6 != 0:
            weight6 = max(0, min(weight7 - 0.5, remainder / number6))

        for stock in range(len(ticker)):
            if int(round(row[stock])) == 9:
                scoreMatrix[quarter + 1][stock + 1] = weight9
            elif int(round(row[stock])) == 8:
                scoreMatrix[quarter + 1][stock + 1] = weight8
            elif int(round(row[stock])) == 7:
                scoreMatrix[quarter + 1][stock + 1] = weight7
            elif int(round(row[stock])) == 6:
                scoreMatrix[quarter + 1][stock + 1] = weight6
            else:
                scoreMatrix[quarter + 1][stock + 1] = 0

    # New directory
    directory_weight = "Random Weights/"
    if not os.path.exists(directory_weight):
        os.makedirs(directory_weight)
    # Count number of files in a folder
    matrix_files = os.listdir(directory_weight)
    number_files = len(matrix_files)

    if number_files == 0:
        title = 'weight matrix (euqal weight)'
    else:
        title = 'weight ' + str(number_files)

    # Write csv file of result matrix after sector selection inside the folder
    with open(directory_weight + title + '.csv', 'w') as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in scoreMatrix:
            writer.writerow(val)


print("Start")
for count in range(iteration):
    start = time.time()
    random_scores()
    end = time.time()
    print("Trial", count + 1, ":", round(end-start, 4), "seconds")

