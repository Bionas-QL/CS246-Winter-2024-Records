import numpy as np

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

np.random.seed(42)
x = np.random.randint(0, 100, size=(15, 10))

U, S, Vh = np.linalg.svd(x, full_matrices=False)
print("The embedding for all 15 documents is:")
print(np.round(U[:, :9] * S[:9], decimals=2))

print("\n")
print("The embedding for all 10 words is:")
print(np.round(S[:9].reshape((9, 1)) * Vh[:9, :], decimals=2))
