import numpy as np
import matplotlib.pyplot as plt


def calc_roc_curve(ground_truth_vector, probability_vector, save_path, plot_on_screen=False):
    """
    Calculate the ROC curve and save to file.
    Assume: positive class is the second class: (0,1) --> positive. (1,0) --> negative
    :param ground_truth_vector: list of ground truths for each sample - i.e: [(0,1), (1,0), ...]
    :param probability_vector:  list of probabilities for each sample - i.e: [(0.4, 0.6), (0.8, 0.1), ...]
    :param save_path: path to save the roc curve
    :param plot_on_screen: if true then plot on screen the ROC curve
    :return:
    """
    tpr_curve = []
    fpr_curve = []
    for decision_treshold in np.arange(0, 1, 0.05):
        tpr = calc_tpr(ground_truth_vector, probability_vector, decision_treshold)
        fpr = calc_fpr(ground_truth_vector, probability_vector, decision_treshold)
        tpr_curve.append(tpr)
        fpr_curve.append(fpr)

    plt.figure()
    lw = 2
    plt.plot(fpr_curve, tpr_curve, color='darkorange', lw=lw)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic ')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    if plot_on_screen:
        plt.show()

    print(tpr_curve)

    print(fpr_curve)


def calc_tpr(ground_truth_vector, probability_vector, decision_treshold):
    """
    Calculate true positive rate.
    :param ground_truth_vector:
    :param probability_vector:
    :param decision_treshold:
    :return:
    """
    # print("Calculating for treshold %f" % decision_treshold)
    num_tp = len([1 for i, _ in enumerate(probability_vector) if probability_vector[i][1] > decision_treshold
                  and ground_truth_vector[i][1] == 1])
    # print("Number of tps:", num_tp)
    num_p = len([1 for e in ground_truth_vector if e[1] == 1])
    # print("Number of positives:", num_p)
    # print(float(num_tp) / float(num_p))
    return float(num_tp) / float(num_p)


def calc_fpr(ground_truth_vector, probability_vector, decision_treshold):
    """
    Calculate false positive rate
    :param ground_truth_vector:
    :param probability_vector:
    :param decision_treshold:
    :return:
    """
    num_fp = len([1 for i, _ in enumerate(probability_vector) if probability_vector[i][1] > decision_treshold
                  and ground_truth_vector[i][1] == 0])

    num_n = len([1 for e in ground_truth_vector if e[1] == 0])

    return float(num_fp) / float(num_n)

if __name__ == "__main__":
    ground_truth = [(0, 1), (1, 0), (0, 1), (1, 0)]
    probs = [(0.3, 0.7), (0.55, 0.45), (0.35, 0.65), (0.1, 0.9)]
    calc_roc_curve(ground_truth_vector=ground_truth, probability_vector=probs, save_path='./b.png', plot_on_screen=True)

    y = np.array([(1,0), (1,0), (0,1), (0,1)])
    scores = np.array([(0.9,0.1), (0.6,0.4), (0.65,0.35), (0.2,0.8)])
    calc_roc_curve(ground_truth_vector=y, probability_vector=scores, save_path='./b.png', plot_on_screen=True)
