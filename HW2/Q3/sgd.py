import numpy as np
import matplotlib.pyplot as plt

k = 20
# Wrong spelling is used deliberately
lamdba = 0.1
MAX_ITER = 200
lr = 0.01

file = open("data/ratings.train.txt", "r")

# Start with the last user
max_user_id = 12
min_user_id = 12
max_item_id = 203
min_item_id = 203

for line in file:
    user_id, item_id, rating = [int(num) for num in line.strip().split("\t")]
    max_user_id = max(max_user_id, user_id)
    min_user_id = min(min_user_id, user_id)
    max_item_id = max(max_item_id, item_id)
    min_item_id = min(min_item_id, item_id)

# One can use min_user_id (=1 in this case) as the starting index, but here for simplicity
# we start with 0

P = np.random.rand(max_user_id + 1, k) * np.sqrt(5 / k)
Q = np.random.rand(max_item_id + 1, k) * np.sqrt(5 / k)
errors = []

for _ in range(MAX_ITER):
    file.seek(0)
    for line in file:
        user_id, item_id, rating = [int(num) for num in line.strip().split("\t")]
        epsilon = 2 * (rating - Q[item_id].dot(P[user_id]))
        new_q = Q[item_id] + lr * (epsilon * P[user_id] - 2 * lamdba * Q[item_id])
        new_p = P[user_id] + lr * (epsilon * Q[item_id] - 2 * lamdba * P[user_id])
        Q[item_id] = new_q
        P[user_id] = new_p

    file.seek(0)
    error = 0
    for line in file:
        user_id, item_id, rating = [int(num) for num in line.strip().split("\t")]
        error += (rating - Q[item_id].dot(P[user_id])) ** 2
    error += lamdba * (np.sum(P ** 2) + np.sum(Q ** 2))
    errors.append(error)

plt.plot(errors)
plt.xlabel("Steps")
plt.ylabel("Errors")
plt.savefig("fig_with_iterations_" + str(MAX_ITER) + ".png")
plt.show()