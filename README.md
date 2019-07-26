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
