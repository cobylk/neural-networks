# All of the weights and biases are initialized to 1 for simplicity

w_1 = 1 # Weight 1
w_2 = 1 # Weight 2
b_1 = 1 # Bias 1
b_2 = 1 # Bias 2

# The neural network will fit a line to any set of data points
data = [[1, -1], [2, 5]]


def forward_propogate(data_point: list[float]) -> float:
    return w_2 * ((w_1 * data_point[0]) + b_1) + b_2

# This loss is squared error
def get_loss(data_point: list[float]) -> float:
    y_hat = forward_propogate(data_point)
    y = data_point[1]
    return (y - y_hat) ** 2

# These get_partial functions return the partial derivative in the function name
# E.g., get_partial_Lw2 returns the partial derivative of the loss (L) with respect to the second weight (w2).
def get_partial_Lw2(data_point: list[float]) -> float:
    x = data_point[0]
    y = data_point[1]
    y_hat = forward_propogate(data_point)
    h_1 = x * w_1 + b_1
    return h_1 * 2 * (y - y_hat)


def get_partial_Lb2(data_point: list[float]) -> float:
    y = data_point[1]
    y_hat = forward_propogate(data_point)
    return 2 * (y - y_hat)


def get_partial_Lw1(data_point: list[float]) -> float:
    x = data_point[0]
    y = data_point[1]
    y_hat = forward_propogate(data_point)
    return x * w_2 * 2 * (y - y_hat)


def get_partial_Lb1(data_point: list[float]) -> float:
    x = data_point[0]
    y = data_point[1]
    y_hat = forward_propogate(data_point)
    return w_2 * 2 * (y - y_hat)


epochs = 1000
learning_rate = 0.01

# This loop trains the network
for i in range(epochs):
    partial_Lw2 = (sum([get_partial_Lw2(data_point) for data_point in data])) / len(data)
    partial_Lb2 = (sum([get_partial_Lb2(data_point) for data_point in data])) / len(data)
    partial_Lw1 = (sum([get_partial_Lw1(data_point) for data_point in data])) / len(data)
    partial_Lb1 = (sum([get_partial_Lb1(data_point) for data_point in data])) / len(data)

    w_2 += learning_rate * partial_Lw2
    b_2 += learning_rate * partial_Lb2
    w_1 += learning_rate * partial_Lw1
    b_1 += learning_rate * partial_Lb1

print(f"w_1: {round(w_1, 5)}, b_1: {round(b_1, 5)}, w_2: {round(w_2, 5)}, b_2: {round(b_2, 5)}")
print(f"Total loss for all data: {round(sum([get_loss(data_point) for data_point in data]), 5)}")
for data_point in data:
    print(f"Input: {data_point[0]}, Output: {round(forward_propogate(data_point), 5)}")
