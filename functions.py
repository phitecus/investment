import math
import numpy as np

def findMax(list):
    if len(list)==0:
        return 0
    else:
        maximum=0
        for value in list:
            if value > maximum:
                maximum = value
        return maximum


def findMaxIndex(list):
    maximum = 0
    index = 0
    for i, value in enumerate(list):
        if value > maximum:
            maximum = value
            index = i
            index = i
    return index


def findMin(list):
    if len(list) == 0:
        return 0
    else:
        minimum = list[0]
        for value in list:
            if value < minimum:
                minimum = value
        return minimum


def findMinIndex(list):
    if len(list)==0:
        return 0
    else:
        minimum = list[0]
    index = 0
    for i, value in enumerate(list):
        if value < minimum:
            minimum = value
            index = i
    return index


def findaverage(list):
    numberofday = 0
    sum = 0
    for i, value in enumerate(list):
        sum += value
        numberofday += 1
    if(numberofday == 0):
        answer=0
    else:
        answer = sum/numberofday
    return answer


def findSD(list):
    mean = findaverage(list)
    newlist = []
    for i, value in enumerate(list):
        newvalue = (value-mean)*(value-mean)
        newlist.append(newvalue)
    if newlist == []:
        answer = 0
    else:
        answer = findaverage(newlist)
    return answer**0.5


#Annual Sharpe
def Sharpe(list):
    if len(list) == 0 or findSD(list) == 0:
        return 0
    else:
        sharpe = (findaverage(list)/(findSD(list)) * (252**(1/2)))
    return sharpe


def replace_blank(array):
    for i in range(len(array)):
        for j in range(len(array[i])):
            if array[i][j]=="":
                array[i][j]=0
    return array


def ranking(list):
    answer = []
    temp = sorted(list,reverse=True)
    for i, value in enumerate(list):
        for j in range(len(temp)):
            if value == temp[j]:
                answer.append(j+1)
    return answer


def replace_na(array):
    for i in range(len(array)):
        for j in range(len(array[i])):
            if array[i][j]=='#N/A N/A' or "":
                array[i][j]=0
    return array


def percentageChange(a, b):
    if a == 0:
        return 0
    else:
        return (b-a)/a*100


def cumulative_list(list):  # Geometric cumulative
    portfolio = 1
    answer = []
    for k in range(len(list)):
        portfolio *= (1 + list[k]/100)
        cumulative = (portfolio - 1) * 100
        answer.append(cumulative)
    return answer


def daily_return_to_cumlist(array):  # Geometric Cumulative
    answer = []
    for i in range(len(array)):
        cum = cumulative_list(array[i])[-1]  # Take the final one
        answer.append(cum)
    return answer


def daily_return_to_annual_pnl(array):  # Arithmetic
    answer = []
    for i in range(len(array)):
        cum = findaverage(array[i]) * 252
        answer.append(cum)
    return answer


def daily_return_to_sharpe(array):
    answer = []
    for i in range(len(array)):
        sharpe = Sharpe(array[i])
        answer.append(sharpe)
    return answer


def logChange(a, b):
    if a != 0 and b != 0:
        return math.log(b/a)*100
    else:
        return 0


def Multiply_entire_row(row, weight):
    for i in range(len(row)):
        A = row[i]
        A *= weight



def WinRate(percentagelist):
    wincount = 0
    losscount = 0
    for i in range(len(percentagelist)):
        if percentagelist[i] > 0:
            wincount += 1
        if percentagelist[i] < 0:
            losscount += 1
    if wincount + losscount == 0:
        winrate = 0
    else:
        winrate = wincount/(wincount+losscount)*100
    return winrate


def MaxConsecutive(percentagelist):
    answer = 0
    list = []
    for i in range(len(percentagelist)):
        if(percentagelist[i] < 0):
            answer+=1
            list.append(answer)
        if(percentagelist[i] >= 0):
            answer = 0
    if list == []:
        key = 0
    else:
        key = findMax(list)
    return key


def is_number(value):
    if isinstance(value, int) or isinstance(value, float):
        return True
    elif any(c.isdigit() for c in value) == True:
        return True


def Check_number(array):  # If the value is not a number, convert it to zero
    for counter in range(len(array)):
        for count in range(len(array[counter])):
            if count == 0:  # Skip the first column (date)
                array[counter][count] = array[counter][count]
            elif is_number(array[counter][count]):
                array[counter][count] = float(array[counter][count])
            else:
                array[counter][count] = 0
    return array


def relative(list):
    answer = []
    first = float(list[0])
    if first == 0:
        k = 0
        while float(list[k]) == 0 and k < len(list) - 1:
            answer.append(1)
            k += 1
        first = float(list[k])
        k -= 1
        while k < len(list)-1:
            k += 1
            if first != 0:
                answer.append(float(list[k])/first)
            else:
                answer.append(1)
    else:
        for i in range(len(list)):
            answer.append(float(list[i])/first)
    return answer


def percentagelist(list):
    answer = []
    for i in range(len(list)-1):
        answer.append(percentageChange(list[i], list[i+1]))
    return answer


def cumulative(list):
    answer = 0
    for i in range(len(list)):
        answer += list[i]
    return answer


def delete_unnecessary_columns(array):
    answer = []
    for row in range(len(array)):
        row_insert = []
        for column in range(len(array[row])):
            if column == 0 or column % 2 == 1:
                row_insert.append(array[row][column])
        answer.append(row_insert)
    return answer


def getYear(item):
    return item[0]


def getQuarter(item):
    return item[1]


def delete_zero(list):
    answer = []
    for element in range(len(list)):
        if list[element] != 0:
            answer.append(list[element])
    return answer


# RSI = 100 - 100 / (1 + RS)
# RS = average gain of up-periods during time frame / average loss of down-periods
# default period is 14 trading days
def get_rsi(list, period):
    rsi = []
    gain = []
    loss = []
    average_gain_list = []
    average_loss_list = []
    for i in range(len(list) - 1):
        change = float(list[i+1]) - float(list[i])
        if change > 0:
            gain.append(change)
            loss.append(0)
        elif change < 0:
            gain.append(0)
            loss.append(-change)
        else:
            gain.append(0)
            loss.append(0)

        if i == period - 1:
            average_gain = findaverage(gain[i - (period - 1):])
            average_gain_list.append(average_gain)
            average_loss = findaverage(loss[i - (period - 1):])
            average_loss_list.append(average_loss)
            if average_loss == 0:
                RS = 0
            else:
                RS = average_gain / average_loss
            RSI = 100 - (100 / (1 + RS))
            rsi.append(RSI)
        elif i > period - 1:
            average_gain = (average_gain_list[-1] * (period - 1) + gain[-1]) / period
            average_gain_list.append(average_gain)
            average_loss = (average_loss_list[-1] * (period - 1) + loss[-1]) / period
            average_loss_list.append(average_loss)
            if average_loss == 0:
                RS = 0
            else:
                RS = average_gain / average_loss
            RSI = 100 - (100 / (1 + RS))
            rsi.append(RSI)
    return rsi


def get_ema(list, period):
    ema = []
    for i in range(len(list)):
        if i == period - 1:
            ema.append(findaverage(list[:period]))
        elif i > period - 1:
            multiplier = 2 / (period + 1)
            EMA = (float(list[i]) - float(ema[-1])) * multiplier + float(ema[-1])
            ema.append(EMA)
    return ema


# Typically, MACD = 12 Day EMA - 26 Day EMA
def get_macd(list, a, b):
    macd = []
    A = get_ema(list, a)
    B = get_ema(list, b)
    for i in range(len(list)):
        if b > a:
            if i > b-2:
                MACD = A[i - (a-1)] - B[i - (b-1)]
                macd.append(MACD)
    return macd


def findzero(list):
    index = 0
    if list[index] == 0 and list[index+1] == 0:
        while index < len(list) and list[index] == 0:
            index += 1
    return index


def change_title(title):
    bool = False
    for i in range(len(title)):
        if title[i] == '/':
            bool = True
    if bool:
        front = title.split('/')[0]
        back = title.split('/')[1]
        answer = front + '-' + back
    else:
        answer = title
    return answer


def portfolio_list(cumulative_list, initial_amount):
    answer = [initial_amount]
    for i in range(len(cumulative_list)):
        portfolio = (1 + cumulative_list[i] / 100) * initial_amount
        answer.append(portfolio)
    return answer


def makePositive(list):
    for int in range(1, len(list)-1):  # Except the first and last row
        if float(list[int]) == 0:
            list[int] = list[int-1]
