import os
import random

NUM_TESTCASES = 10000
os.makedirs("Testcase", exist_ok=True)


def save_testcase(index, p, prob_connect, causal_order):
    with open(f"Testcase/{index}.txt", "w", encoding="utf-8") as f:
        f.write(f"p={p}\n")
        f.write(f"probConnect={prob_connect}\n")
        f.write("causalOrder=" + " ".join(map(str, causal_order)) + "\n")


def random_order(p):
    order = list(range(1, p + 1))
    random.shuffle(order)
    return order


case_index = 1

#small and edge cases
save_testcase(case_index, 1, 0.0, [1])
case_index += 1

save_testcase(case_index, 1, 1.0, [1])
case_index += 1

save_testcase(case_index, 2, 0.0, [1, 2])
case_index += 1

save_testcase(case_index, 2, 1.0, [1, 2])
case_index += 1

save_testcase(case_index, 2, 0.0, [2, 1])
case_index += 1

save_testcase(case_index, 2, 1.0, [2, 1])
case_index += 1

save_testcase(case_index, 3, 0.0, [1, 2, 3])
case_index += 1

save_testcase(case_index, 3, 1.0, [1, 2, 3])
case_index += 1

save_testcase(case_index, 3, 0.5, [1, 2, 3])
case_index += 1

save_testcase(case_index, 3, 0.5, [3, 2, 1])
case_index += 1

save_testcase(case_index, 3, 0.5, [2, 1, 3])
case_index += 1

save_testcase(case_index, 4, 0.0, [4, 3, 2, 1])
case_index += 1

#llm
#some fixed testcases
fixed_cases = [
    (5, 0.2, [1, 2, 3, 4, 5]),
    (5, 0.5, [1, 2, 3, 4, 5]),
    (5, 1.0, [1, 2, 3, 4, 5]),
    (5, 0.2, [5, 4, 3, 2, 1]),
    (5, 0.5, [5, 4, 3, 2, 1]),
    (5, 1.0, [5, 4, 3, 2, 1]),
    (5, 0.3, [3, 1, 5, 2, 4]),
    (5, 0.7, [3, 1, 5, 2, 4]),
    (6, 0.2, [1, 2, 3, 4, 5, 6]),
    (6, 0.5, [1, 2, 3, 4, 5, 6]),
    (6, 0.8, [1, 2, 3, 4, 5, 6]),
    (6, 0.2, [6, 5, 4, 3, 2, 1]),
    (6, 0.5, [6, 5, 4, 3, 2, 1]),
    (6, 0.8, [6, 5, 4, 3, 2, 1]),
    (6, 0.4, [2, 5, 1, 6, 3, 4]),
    (6, 0.9, [2, 5, 1, 6, 3, 4]),
    (8, 0.3, [1, 2, 3, 4, 5, 6, 7, 8]),
    (8, 0.3, [8, 7, 6, 5, 4, 3, 2, 1]),
]

for p, prob_connect, causal_order in fixed_cases:
    save_testcase(case_index, p, prob_connect, causal_order)
    case_index += 1


#same order and different probabilities
for prob_connect in [0.0, 0.1, 0.3, 0.5, 0.8, 1.0]:
    save_testcase(case_index, 8, prob_connect, [8, 2, 6, 1, 4, 5, 3, 7])
    case_index += 1

for prob_connect in [0.0, 0.1, 0.3, 0.5, 0.8, 1.0]:
    save_testcase(case_index, 10, prob_connect, [10, 4, 8, 1, 7, 3, 2, 9, 5, 6])
    case_index += 1

for prob_connect in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    save_testcase(case_index, 12, prob_connect, [12, 1, 10, 3, 8, 5, 6, 2, 11, 4, 9, 7])
    case_index += 1


#same size and different orders
for p in [3, 4, 5, 6, 8, 10, 12, 15]:
    save_testcase(case_index, p, 0.3, list(range(1, p + 1)))
    case_index += 1

for p in [3, 4, 5, 6, 8, 10, 12, 15]:
    save_testcase(case_index, p, 0.7, list(range(p, 0, -1)))
    case_index += 1

for p in [4, 6, 8, 10]:
    save_testcase(case_index, p, 0.5, random_order(p))
    case_index += 1


#random cases for the rest
while case_index <= NUM_TESTCASES:
    p = random.randint(3, 15)
    prob_connect = random.choice([0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0])
    causal_order = random_order(p)

    save_testcase(case_index, p, prob_connect, causal_order)
    case_index += 1