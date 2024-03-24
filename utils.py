import json
from sklearn.metrics import accuracy_score


def calculate_metrics(targets_list, predictions_list):
    accuracy = accuracy_score(targets_list, predictions_list)

    cer = 0
    for i, (target, prediction) in enumerate(zip(targets_list, predictions_list)):
        distance = levenshtein_distance(target, prediction)
        cer += distance

    return accuracy * 100, cer / len(targets_list) * 100


# chat give me that - check it
def levenshtein_distance(str1, str2):
    m = len(str1)
    n = len(str2)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if str1[i - 1] == str2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,
                           dp[i][j - 1] + 1,
                           dp[i - 1][j - 1] + cost)

    max_distance = max(m, n)
    distance = ((dp[m][n]) / max_distance)
    return distance


def read_config(path):
    with open(path, "r") as file:
        data = json.load(file)
    return data
