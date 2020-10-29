import numpy as np
import random as rm

# The statespace
states = ["Sleep","Icecream","Run"]

# Transition Matrix
TransName = [["SS","SR","SI"],["RS","RR","RI"],["IS","IR","II"]]
TransMat = [[0.2,0.6,0.2],[0.1,0.6,0.3],[0.2,0.7,0.1]]
assert(sum(TransMat[0])+sum(TransMat[1])+sum(TransMat[1]) == 3)

# Forecasting
def Fcst(days):
    currState = "Sleep"
    print("Start state: " + currState)
    states = [currState]
    i = 0
    p = 1
    while i != days:
        if currState == "Sleep":
            change = np.random.choice(TransName[0],replace=True,p=TransMat[0])
            if change == "SS": p = p * 0.2; states.append("Sleep"); pass
            elif change == "SR": p = p * 0.6; currState = "Run"; states.append("Run")
            else: p = p * 0.2 ; currState = "Icecream"; states.append("Icecream")
        elif currState == "Run":
            change = np.random.choice(TransName[1],replace=True,p=TransMat[1])
            if change == "RR": p = p * 0.5; states.append("Run"); pass
            elif change == "RS": p = p * 0.2; currState = "Sleep"; states.append("Sleep")
            else: p = p * 0.3; currState = "Icecream"; states.append("Icecream")
        elif currState == "Icecream":
            change = np.random.choice(TransName[2],replace=True,p=TransMat[2])
            if change == "II": p = p * 0.1; states.append("Icecream"); pass
            elif change == "IS": p = p * 0.2; currState = "Sleep"; states.append("Sleep")
            else: p = p * 0.7; currState = "Run"; states.append("Run")
        i += 1  
    print("Possible states: " + str(states))
    print("End state after "+ str(days) + " days: " + currState)
    print("Probability of the possible sequence of states: " + str(p))


# Testing
Fcst(2)





