import json
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np


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


def create_plot(train_data, test_data, plot_name):
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    plt.figure(figsize=(8, 6))
    plt.plot((range(len(train_data))), train_data, label="Train set", linewidth=2)
    plt.plot((range(len(test_data))), test_data, label="Test set", linewidth=2)
    plt.legend(loc="lower right")
    plt.grid()
    plt.xlabel('Epochs')
    plt.ylabel(f'{plot_name}')
    if plot_name != "Loss value":
        plt.yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    return plt


def save_model_results(path, lists):
    list_train_loss = lists["train_loss"]
    list_train_accuracy = lists["train_accuracy"]
    list_train_cer = lists["train_cer"]

    list_test_loss = lists["test_loss"]
    list_test_accuracy = lists["test_accuracy"]
    list_test_cer = lists["test_cer"]

    loss_plt = create_plot(list_train_loss, list_test_loss, "Loss value")
    loss_plt.savefig(f"{path}/lossplt.png")
    accuracy_plt = create_plot(list_train_accuracy, list_test_accuracy, "Accuracy [%]")
    accuracy_plt.savefig(f"{path}/accuracy.png")
    cer_plt = create_plot(list_train_cer, list_test_cer, "CER [%]")
    cer_plt.savefig(f"{path}/cer.png")


def read_config(path):
    with open(path, "r") as file:
        data = json.load(file)
    return data
