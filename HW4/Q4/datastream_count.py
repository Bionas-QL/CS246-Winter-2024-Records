import matplotlib.pyplot as plt
import numpy as np


def hash_fun(a, b, p, n_buckets, x):
    y = x % p
    hash_val = (a * y + b) % p
    return hash_val % n_buckets



hash_params = []
file = open("data/hash_params.txt", "r")
for line in file:
    hash_params.append((int(line.strip().split("\t")[0]), int(line.strip().split("\t")[1])))
file.close()

num_of_hash_funcs = len(hash_params)
n_buckets = 10 ** 4
hash_count = np.zeros((num_of_hash_funcs, n_buckets))
p = 123457

file = open("data/words_stream.txt", "r")
for line in file:
    word = int(line.strip())
    for i in range(len(hash_params)):
        hash_count[i, hash_fun(hash_params[i][0], hash_params[i][1], p, n_buckets, word)] += 1
file.close()
stream_length = np.sum(hash_count[0])

rel_errors = []
freq_list = []
file = open("data/counts.txt", "r")
for line in file:
    word = int(line.strip().split("\t")[0])
    freq = int(line.strip().split("\t")[1])
    approx_freq = hash_count[0, hash_fun(hash_params[0][0], hash_params[0][1], p, n_buckets, word)]
    for i in range(1, len(hash_params)):
        approx_freq = min(approx_freq, hash_count[i, hash_fun(hash_params[i][0], hash_params[i][1], p, n_buckets, word)])
    rel_error = (approx_freq - freq) / freq
    rel_errors.append(rel_error)
    freq_list.append(freq)
    # if rel_error < 1:
    #    print("For word with id ", word, " the relative error is ", rel_error)
file.close()

freq_list = np.array(freq_list) / stream_length
rel_errors = np.array(rel_errors)

plt.figure()
plt.loglog(freq_list, rel_errors, 'o')
plt.xlabel("Exact Word Frequency")
plt.ylabel("Relative Error")
plt.title("5 hash functions and 10000 buckets")
plt.grid()
plt.savefig("result.png")
plt.show()

