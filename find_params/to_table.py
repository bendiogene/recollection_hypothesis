import os
import re

# list of paths to the files with the results
FILES_LIST = ["./res_all_spikes_same_thresholds.txt"]
# path of the output file
OUTPUT_PATH = "./"


def print_to_file(results, order_by=None):
    if order_by == "avg":
        sorted_results = sorted(results, key=lambda k: 0.5 * sum(results[k]), reverse=True)
        filename = "res_avg.tex"
    elif order_by == "train":
        sorted_results = sorted(results, key=lambda k: results[k][0], reverse=True)
        filename = "res_train.tex"
    elif order_by == "test":
        sorted_results = sorted(results, key=lambda k: results[k][1], reverse=True)
        filename = "res_test.tex"

    with open(os.path.join(OUTPUT_PATH, filename), "w") as out_file:
        out_file.write(
            '\\documentclass{article}\n\\usepackage{longtable}\n\\usepackage[utf8]{inputenc}\n\\begin{document}\n\\section{Results}\n\\begin{longtable}{|c|c|c|c|c|c|c|c|c|c|c|}\n\\hline\nN iter & $\\alpha^+$ & $\\alpha^-$ & TR\\_learn & n\\_spikes & pool\\_w & random\\_init & tr\_l1 & tr\_l2 & train & test \\\ \\hline\n')
        for item in sorted_results:
            out_file.write('{}&{}&{}&{}&{}&{}&{}&{}&{}&{}&{}\\\ \\hline\n'.format(*item))
        out_file.write('\n\\end{longtable}\n\\end{document}\n')


if __name__ == "__main__":
    results = {}

    for res_file in FILES_LIST:
        with open(res_file) as f:
            for line in f:
                if line.startswith("ITER"):
                    matches = re.findall(
                        'ITER_TRAIN: (.*?), APLUS: (.*?),AMINUS: (.*?),TR_learning: (.*?),SPIKES_T0_CONSIDER: (.*?),POOLING_W: (.*?),RANDOM_INIT: (.*?),TR_L1: (.*?),TR_L2: (.*?),CLASS_TR: (.*)',
                        line, re.DOTALL)
                    print(matches)
                    #next(f)
                    #next(f)
                    n_iter, aplus, aminus, tr_learning, n_spikes, pool_w, random_init, tr_l1, tr_l2, class_tr  = matches[0]
                    # get train score
                    train_score = round(float(re.findall('p: (.*)\n', next(f), re.DOTALL)[0]), 4)
                    # get test score
                    test_score = round(float(re.findall('p: (.*)\n', next(f), re.DOTALL)[0]), 4)
                    # consider the result only if both train and test are > 0.9
                    if True:#aplus > aminus and n_spikes=="-1":
                    #if aplus > aminus and train_score > 0.9 and test_score > 0.9:
                        results[(
                            n_iter, aplus, aminus, tr_learning, n_spikes, pool_w, random_init, tr_l1, tr_l2, train_score,
                            test_score)] = (
                            train_score, test_score)
        print(results)

    print_to_file(results, "avg")
    print_to_file(results, "train")
    print_to_file(results, "test")
