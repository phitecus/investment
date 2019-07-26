# Overview

* Sector Selection

* Stock Selection

* Market Timing


## 1. Sector Selection

### Update files in “update” folder, using Bloomberg

There are 4 csv files in the folder to update for sector selection:
“earnings_growth_bloomberg.csv”, “HSCI_bloomberg.csv”, “market
cap_bloomberg.csv”, and “prices_Bloomberg.csv”

Except for HSCI, use macro to copy and paste functions every other
column and then data will be updated automatically

After updating from Bloomberg, Run “Backtest_sector_selection.py”:

It will automatically run “Update.py” and “Sector_Selection.py”

### Choose best 4 sectors out of 11 Sectors semi-annually

Consumer Discretionary, Consumer Staples, Energy, Financials, Health Care,
Industrials, Information Technology, Materials, Real Estate,
Telecommunication Services, Utilities

Each stock is assigned to one of these sectors

There are 4 python files (“Update.py”, “functions.py”,
“Sector_Selection.py”, “Backtest_sector_selection.py”)

4 csv files are required to be imported (“earnings growth.csv”, “market
cap.csv”, “HSCI.csv”, and “prices.csv”)

“earnings growth.csv”, “market cap.csv” are represented in every quarter,
and “prices.csv” and “HSCI.csv” are represented in every trading day.


#### Step 1. Weighted Earnings Growth

```  
Weighted Earnings Growth = Earnings Growth / Total Earnings Growth of the sector
```  

This step will be automatically carried out in Update.py

It is represented in every quarter (1985Q1 to 2018Q1)

#### Step 2. Rank sectors based on Q-2 weighted earnings growth

In the “Sector_Selection.py” program, it will import the weighted
“earnings growth.csv” file

It will skip the data before 1998Q4 due to lack of data

The program will sum up the weighted earnings growth of each
sector and rank them accordingly for each time period (every 2Q)

The python program will create the result folder and save the
rankings with each time period inside the folder (“Ranking based on
Weighted Earnings Growth.csv”)

#### Step 3.Ranking [1 and 0]

This is for easier calculation


#### Step 4. Back Test

After choosing the sectors every 2 quarter, we allocate resources
accordingly (I have done equal weight in this part, not based on
market cap)

Set the initial amount to 1M and sector selection = 4

Then run the Back Test: it will automatically run “Update.py” and
“Sector_Selection.py”


#### Step 5. Preparation

Automatically read the related CSV files: “Ranking [1 and 0].csv” and
“prices.csv”

It will initialize the result matrix with correct headers and zeros.

Prices with no data will be also set to be zero.


#### Step 6. Loop through time index (y direction)

Read the time index and find the corresponding quarter to get the
ranking and signal.

For each time period, loop through stocks

#### Step 7. Loop through stock (x direction)

Calculate the percentage change of price of the stock

Return = Percentage change of price * signal (signal = 1 or 0)

Save the return value in the corresponding result matrix

While looping through stocks, count the number of stocks to be
invested (if signal is 1 and price is not 0)

#### Step 8. Calculate daily & cumulative return

After looping one entire row, calculate the daily return: sum the
percentage change in the row and divide it by the stock count (equal
weight)

Equal weight: Daily return = Sum(return) / stock count

Market cap weight: Daily return = sum(return*weight)

Store the daily return in “daily_return” list.

Also calculate cumulative return.

Then go to the next iteration (next day)

Save the results in Sector Selection Result Folder.

<img src="./Sector Selection Result/HSCI Sector Selection.png" width="450">


## 2. Stock Selection

### How to Update

Update 7 csv files using Bloomberg. So far data until 2017Q4 have
been updated, so only the data after 2017Q4 will be required.

After updating from Bloomberg, copy and paste into “Stock Selection
folder” inside “update folder” (relevant python folder)

Then run “Backtest_stock_selection.py”

#### 1. Update_stock_selection.py

The purpose of this program is to preprocess the data downloaded
from Bloomberg.

It will import 7 current csv files, and other new 7 csv files in
“update\Stock Selection” folder (~_Bloomberg.csv)

If new csv files have more up-to-date data, this program will append
those data into current csv files with the right format for future use.

#### 2. Stock_Score.py

The purpose of this python file is to output Piotroski F Score (0~9) for
every quarter (HK: semiannually)

It will import the necessary fundamental values to calculate scores

The result will be saved in “Stock Picking Result” folder as ‘score
matrix.csv’

Another result with Sector screening will also be saved in “Stock
Picking Result” folder as ‘score matrix after sector selection.csv’

#### 3. weight1.py & weight2.py

The purpose of these files are to simply output weights in excel files
for each weight scheme.

Each weight.py automatically runs ‘Stock_Score.py’, saves the scores
and uses the results to assign weights to each stock for each time
period.

The corresponding results will also be saved in ‘Stock Picking Result’
folder as ‘weight 1.csv’ and ‘weight 2.csv’

#### 4. Backtest_stock_selection.py

Finally, this program will automatically run all the previous programs

Plus, it will draw relevant graphs and save them in “Sector Selection
Result”

Daily Cumulative Return for each weight scheme, combined graph,
Semiannual Return graph, and results with each weighting scheme

<img src="./Stock Picking Result/HSCI Semiannual Return Combined.png" width="450">

<img src="./Stock Picking Result/HSCI Stock Selection Combined.png" width="450">



## 3. Market Timing

### Optimization.py

Generate random parameters for each HSCI stock
with annual return and Sharpe Ratio inside
“Optimization” folder (input: number of iteration)

### market_timing.py

Function that generate each signal of technical
indicators, trading cost, cumulative return with
given parameters according to time series for each
stock inside “Market Timing Result” folder

### Backtest_market_timing.py

With the inputs of year and initial amount: choose
the parameters that generate highest return for
each stock and generate various graphs and
statistics inside “Market Timing Performance”
folder

<img src="./Market Timing Performance/HSCI Market Timing from 2008.png" width="450">



