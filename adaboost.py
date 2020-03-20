from matplotlib import pyplot as plt
import numpy as np
from process_data import parse_data
import matplotlib.patches as mpatches

np.random.seed(7)


def run_adaboost(X_train, y_train, T):
    """
    Returns: 

        hypotheses : 
            A list of T tuples describing the hypotheses chosen by the algorithm. 
            Each tuple has 3 elements (h_pred, h_index, h_theta), where h_pred is 
            the returned value (+1 or -1) if the count at index h_index is <= h_theta.

        alpha_vals : 
            A list of T float values, which are the alpha values obtained in every 
            iteration of the algorithm.
    """
    hypotheses = list()
    alpha_vals = list()
    # initialize
    m = len(X_train)
    D = [1 / m] * m
    # boost
    for t in range(T):
        ht = weak_learner(D, X_train, y_train)
        hypotheses.append(ht)
        error = compute_error(D, X_train, y_train, ht)
        at = 0.5 * np.log(1 / error - 1)
        alpha_vals.append(at)
        zt = compute_zt(D, X_train, y_train, ht, at)
        for i in range(m):
            D[i] = D[i] * np.exp(-1 * at * y_train[i] * compute_ht(ht, X_train[i])) / zt

    return hypotheses, alpha_vals


def weak_learner(D, X_train, y_train):
    best_loss = np.inf
    for j in range(len(X_train[0])):
        # sort S using the j coordinate
        sorted_S = [(i, X_train[i][j], y_train[i], D[i]) for i in range(len(X_train))]
        sorted_S.sort(key=lambda tup: tup[1])
        # add possible last theta
        sorted_S.append((0, sorted_S[len(sorted_S) - 1][1] + 1, 0, 0))
        # loss_1 = sum (Di for yi==1)
        loss_1 = 0
        for i in range(len(sorted_S) - 1):
            # yi==1
            if sorted_S[i][2] == 1:
                # loss_1 += Di
                loss_1 += sorted_S[i][3]
        loss_2 = 1 - loss_1
        ###
        # check for h_pread = 1
        if loss_1 < best_loss:
            best_loss = loss_1
            best_theta = sorted_S[0][1] - 1
            best_index = j
            best_pred = 1

        for i in range(len(X_train)):
            # loss_1 = loss_1 -yi*Di
            loss_1 = loss_1 - sorted_S[i][2] * sorted_S[i][3]
            if loss_1 < best_loss and sorted_S[i][1] != sorted_S[i + 1][1]:
                best_loss = loss_1
                best_theta = 0.5 * (sorted_S[i][1] + sorted_S[i + 1][1])
                best_index = j
                best_pred = 1

        # check for h_pread = -1
        if loss_2 < best_loss:
            best_loss = loss_2
            best_theta = sorted_S[0][1] - 1
            best_index = j
            best_pred = -1

        for i in range(len(X_train)):
            # loss_2 = loss_2 +yi*Di
            loss_2 = loss_2 + sorted_S[i][2] * sorted_S[i][3]
            if loss_2 < best_loss and sorted_S[i][1] != sorted_S[i + 1][1]:
                best_loss = loss_2
                best_theta = 0.5 * (sorted_S[i][1] + sorted_S[i + 1][1])
                best_index = j
                best_pred = -1

    ht = (best_pred, best_index, best_theta)
    return ht


def compute_ht(ht, xi):
    h_pred = ht[0]
    h_index = ht[1]
    h_theta = ht[2]
    if xi[h_index] <= h_theta:
        return h_pred
    return h_pred * -1


def compute_error(D, X_train, y_train, ht):
    m = len(X_train)
    error = 0
    for i in range(m):
        if y_train[i] != compute_ht(ht, X_train[i]):
            error += D[i]
    return error


def compute_zt(D, X_train, y_train, ht, at):
    m = len(X_train)
    zt = 0
    for i in range(m):
        zt += D[i] * np.exp(-1 * at * y_train[i] * compute_ht(ht, X_train[i]))
    return zt


##############################################
def sign_func(hypotheses, alpha_vals, xi):
    classifier = list()
    for i in range(len(hypotheses)):
        classifier.append(alpha_vals[i] * compute_ht(hypotheses[i], xi))
    if sum(classifier) >= 0:
        return 1
    else:
        return -1


def calc_error(hypotheses, alpha_vals, X_set, y_set):
    error = 0
    for i in range(len(X_set)):
        if sign_func(hypotheses, alpha_vals, X_set[i]) != y_set[i]:
            error += 1
    return error / len(X_set)


def calc_exp_error(hypotheses, alpha_vals, X_set, y_set):
    exp_error = 0
    for i in range(len(X_set)):
        classifier = list()
        for j in range(len(hypotheses)):
            classifier.append(alpha_vals[j] * compute_ht(hypotheses[j], X_set[i]))
        exp_error += np.exp(-y_set[i] * sum(classifier))
    return exp_error / len(X_set)

def main():
    data = parse_data()
    if not data:
        return
    (X_train, y_train, X_test, y_test, vocab) = data

    T = 10
    hypotheses, alpha_vals = run_adaboost(X_train, y_train, T)

    print("weak classifiers for 10 iterations:")
    for i in range(T):
        print("h{} = {}, Word = \"{}\" ".format(i, hypotheses[i], vocab[hypotheses[i][1]]) + "weight is {0:.2f}".format(alpha_vals[i]))

    
    train_error = list()
    test_error = list()
    exp_train_error = list()
    exp_test_error = list()
    for t in range(T):
        train_error.append(calc_error(hypotheses[:t + 1], alpha_vals[:t + 1], X_train, y_train))
        test_error.append(calc_error(hypotheses[:t + 1], alpha_vals[:t + 1], X_test, y_test))
        exp_train_error.append(calc_exp_error(hypotheses[:t + 1], alpha_vals[:t + 1], X_train, y_train))
        exp_test_error.append(calc_exp_error(hypotheses[:t + 1], alpha_vals[:t + 1], X_test, y_test))
    plt.plot([t for t in range(T)], train_error, color='r', marker='x')
    plt.plot([t for t in range(T)], test_error, color='b', marker='o')
    plt.title('Error as func of T')
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    red_patch_zero = mpatches.Patch(color='red', label='Training error')
    blue_patch_zero = mpatches.Patch(color='blue', label='Test error')
    plt.legend(handles=[red_patch_zero, blue_patch_zero])
    plt.show()
    plt.clf()

    plt.plot([t for t in range(T)], exp_train_error, color='r', marker='x')
    plt.plot([t for t in range(T)], exp_test_error, color='b', marker='o')
    plt.title('exponential loss as func of T')
    plt.xlabel("Iteration")
    plt.ylabel("exp loss")
    red_patch_exp = mpatches.Patch(color='red', label='Training exp_error')
    blue_patch_exp = mpatches.Patch(color='blue', label='Test exp_error')
    plt.legend(handles=[red_patch_exp, blue_patch_exp])
    plt.show()


if __name__ == '__main__':
    main()
