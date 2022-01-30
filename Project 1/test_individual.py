import numpy as np
from nn_functions import *
from network_structures import *
from testing_network import *

a_func = sigmoid
da_func = d_sigmoid

w_11 = 0
w_21 = 3
w_31 = 6
w_12 = 1
w_22 = 4
w_32 = 7
w_13 = 2
w_23 = 5
w_33 = 8

weights = np.array([
    [w_11, w_12, w_13],
    [w_21, w_22, w_23],
    [w_31, w_32, w_33]
])

z1 = 0.4
z2 = 0.6
z3 = 0.8

proper_diagonal = np.array([
    [z1, 0, 0],
    [0, z2, 0],
    [0, 0, z3]
])

correct_result = np.array([
    [w_11 * z1 * (1 - z1), w_21 * z1 * (1 - z1), w_31 * z1 * (1 - z1)],
    [w_12 * z2 * (1 - z2), w_22 * z2 * (1 - z2), w_32 * z2 * (1 - z2)],
    [w_13 * z3 * (1 - z3), w_23 * z3 * (1 - z3), w_33 * z3 * (1 - z3)]
])

def get_J_zy():
    inputs = np.array([z1,z2,z3]).reshape(3,1)
    J_sum = np.diag(da_func(inputs[:, 0]))
    # print(J_sum)
    # Einsum corresponds to inner product of J_sum * weights.T
    a = np.einsum('ij,kj->ik', J_sum, weights)
    print(a)
    print(correct_result)

# AIGHT ABOVE FUCKING FUNCTION WORKS, CHECK NEXT

y1 = 0.4
y2 = 0.3
y3 = 0.7

Y = np.array([y1, y2, y3]).reshape(3,1)
# Z = inputs

correct_result2 = np.array([
    [y1 * z1 * (1 - z1), y1 * z2 * (1 - z2), y1 * z3 * (1 - z3)],
    [y2 * z1 * (1 - z1), y2 * z2 * (1 - z2), y2 * z3 * (1 - z3)],
    [y3 * z1 * (1 - z1), y3 * z2 * (1 - z2), y3 * z3 * (1 - z3)]
])



def get_J_zw(prv_inputs, inputs):
    diag_J_sum = da_func(inputs[:, 0])
    b = np.einsum('i,j->ij', prv_inputs[:,0], diag_J_sum)
    print(b)
    print(correct_result2)


# THIS FUNCTION ALSO FUCKING WORKS, WTF. OK CHECK LAST ONE


# TEST J LOSS HERE

z1 = 0.4
z2 = 0.6
z3 = 0.8

t1 = 0.2
t2 = 0.8
t3 = 0.6

l_func = mse
dl_func = d_mse

correct_result3 = np.array([
    [2/3 * (z1 - t1), 2/3 *  (z2 - t2), 2/3 * (z3 - t3)]
])

predictions = np.array([z1,z2,z3]).reshape(3,1)
targets = np.array([t1,t2,t3]).reshape(3,1)

def get_J_lz():
    predictions = np.array([z1,z2,z3]).reshape(3,1)
    targets = np.array([t1,t2,t3]).reshape(3,1)
    return dl_func(predictions, targets).T

print(correct_result3)

# LOSS JACOBIAN WORKS AS INTENDED

# Test the dot-product of J^L_Z . J^Z_Y = J^L_Y

w_11 = 0
w_21 = 3
w_12 = 1
w_22 = 4
w_13 = 2
w_23 = 5

weights = np.array([
    [w_11, w_12, w_13],
    [w_21, w_22, w_23],
])

z1 = 0.4
z2 = 0.6
z3 = 0.8

correct_result = np.array([
    [w_11 * z1 * (1 - z1), w_21 * z1 * (1 - z1)],
    [w_12 * z2 * (1 - z2), w_22 * z2 * (1 - z2)],
    [w_13 * z3 * (1 - z3), w_23 * z3 * (1 - z3)]
])

def get_J_lz():
    predictions = np.array([z1,z2,z3]).reshape(3,1)
    targets = np.array([t1,t2,t3]).reshape(3,1)
    return dl_func(predictions, targets).T

def get_J_zy():
    inputs = np.array([z1,z2,z3]).reshape(3,1)
    J_sum = np.diag(da_func(inputs[:, 0]))
    # print(J_sum)
    # Einsum corresponds to inner product of J_sum * weights.T
    a = np.einsum('ij,kj->ik', J_sum, weights)
    return a

test_result = np.dot(get_J_lz(), get_J_zy())

correct = np.array([
    [2/3 * (z1-t1) * z1 * (1-z1) * w_11 + 2/3 * (z2-t2) * z2 * (1-z2) * w_12 + 2/3 * (z3-t3) * z3 * (1-z3) * w_13],
    [2/3 * (z1-t1) * z1 * (1-z1) * w_21 + 2/3 * (z2-t2) * z2 * (1-z2) * w_22 + 2/3 * (z3-t3) * z3 * (1-z3) * w_23]
])

print(f"Correct: {correct}")
print(f"Test: {test_result}")

# WELL SHIT THIS IS CORRECT AS WELL

# Final jacobian function to test, is the weight-jacobian
# For some reason, this shit doesn't work properly

x1 = 0.4
x2 = 0.1
x3 = 0.8

y1 = 0.4
y2 = 0.3

z1 = 0.4

X = np.array([x1,x2,x3]).reshape(3,1)
Y = np.array([y1,y2]).reshape(2,1)
Z = np.array([z1]).reshape(1,1)

w11 = 0.2
w21 = 0.1

Z_weights = np.array([w11, w21]).reshape(2,1)

v11 = 0.1
v12 = 0.2
v21 = 0.5
v22 = 0.4
v31 = 0.3
v32 = 0.2

Y_weights = np.array([
    [v11, v12],
    [v21, v22],
    [v31, v32]
])

correct_result4 = np.array([
    [z1*(1-z1)*x1*y1*(1-y1)*w11, z1*(1-z1)*x1*y2*(1-y2)*w21],
    [z1*(1-z1)*x2*y1*(1-y1)*w11, z1*(1-z1)*x2*y2*(1-y2)*w21],
    [z1*(1-z1)*x3*y1*(1-y1)*w11, z1*(1-z1)*x3*y2*(1-y2)*w21]
])


def check():
    pass

if __name__ == "__main__":
    # test_backprop()
    # print(Z_weights)
    # print(Y_weights)
    # print(correct_result3)
    print(get_J_lz())
