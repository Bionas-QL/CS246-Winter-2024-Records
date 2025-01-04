import numpy as np

file = open("data/user-shows.txt", "r")
R = np.zeros((9985, 563))
row = 0
for line in file:
    R[row] = np.array([int(num) for num in line.strip().split(" ")])
    row += 1
file.close()

file = open("data/shows.txt")
show_names = []
for line in file:
    show_names.append(line.strip())
file.close()

P = np.diag(np.sum(R, axis=1))
Q = np.diag(np.sum(R, axis=0))

user_gamma = np.diag(1 / np.sqrt(np.sum(R, axis=1))) @ R @ R.T @ np.diag(1 / np.sqrt(np.sum(R, axis=1))) @ R
Alex_rs = user_gamma[499, 0:100].tolist()
Alex_rs = [(Alex_rs[i], -i) for i in range(len(Alex_rs))]
Alex_rs.sort(reverse=True)
print("The shows given by the highest five user-user recommendation scores for Alex are:")
for i in range(5):
    print(show_names[-Alex_rs[i][1]])

item_gamma = R @ np.diag(1 / np.sqrt(np.sum(R, axis=0))) @ R.T @ R @ np.diag(1 / np.sqrt(np.sum(R, axis=0)))
Alex_rs = item_gamma[499, 0:100].tolist()
Alex_rs = [(Alex_rs[i], -i) for i in range(len(Alex_rs))]
Alex_rs.sort(reverse=True)
print("The shows given by the highest five item-item recommendation scores for Alex are:")
for i in range(5):
    print(show_names[-Alex_rs[i][1]])