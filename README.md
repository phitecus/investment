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

