import numpy as np

def predict_demand(x, w, b):

    return w * x + b


x = 1000
w = 0.05
b = 20

demand = predict_demand(x, w, b)
print("Predicted Demand:", demand)

