import itertools
import os

import pandas as pd
import tabulate as tb
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.feature_extraction as skf


def get_deck(cabin):
    if pd.isna(cabin):
        return np.nan
    else:
        return re.sub("[^a-zA-Z]", "", cabin)[0]


def exercise1():
    data = pd.read_csv(os.path.join("all", "train.csv"), sep=",")
    data["Deck"] = data.apply(lambda row: get_deck(row["Cabin"]), axis=1)
    for col in ["Deck", "Sex"]:
        data[col] = pd.Categorical(data[col]).codes
    data = data.fillna({"Age": data["Age"].mean()}).drop(columns=["Name", "Ticket", "Embarked", "Fare", "Cabin"])
    for col in ["Deck"]:
        data[col] = data[col].replace(-1, data.loc[data[col] != -1][col].mode()[0])
    data.to_csv("w2.csv", sep=",", index=False)
    data.to_json("w2.json", orient="records")


def process_group(group, group_name, ctg, num):
    print("Average %s" % group_name)
    for clmn in ctg:
        if not os.path.exists(group_name):
            os.makedirs(group_name)
        print "For %s value is %s" % (clmn, group[clmn].mode()[0])
        group[clmn].value_counts(sort=False).plot(kind="bar", rot=0)
        plt.savefig(os.path.join(group_name, "%s.png" % clmn))
        plt.close()
    for clmn in num:
        print "For %s value is %s" % (clmn, group[clmn].median())
        group.boxplot(column=clmn)
        plt.savefig(os.path.join(group_name, "%s.png" % clmn))
        plt.close()


def exercise2():
    data = pd.read_csv(os.path.join("all", "train.csv"), sep=",")
    data["Deck"] = data.apply(lambda row: get_deck(row["Cabin"]), axis=1)
    categorical = ["Pclass", "Sex", "SibSp", "Parch", "Deck"]
    numerical = ["Age", "Fare"]
    for col in categorical:
        print "For %s value is %s" % (col, data[col].mode()[0])
        data[col] = data[col].fillna(data[col].mode()[0])
    for col in numerical:
        print "For %s value is %s" % (col, data[col].median())
        data[col] = data[col].fillna(data[col].mean())
    print
    process_group(data.loc[data["Survived"] == 1], "survivor", categorical, numerical)
    process_group(data.loc[data["Survived"] == 0], "non-survivor", categorical, numerical)


def get_highest_ranks(row):
    return tuple(row)


if __name__ == "__main__":
    # exercise1()
    # exercise2()
    with open("tfidf.txt", "w") as tfidf_f:
        with open("pos.txt", "r") as f:
            tfidf = skf.text.TfidfVectorizer(stop_words="english")
            tfidf_matrix = tfidf.fit_transform(f)
            feature_names = np.array(tfidf.get_feature_names())
            rows, cols = tfidf_matrix.shape
            for i in range(0, rows):
                row = tfidf_matrix.getrow(i).A[0]
                sorted_args = np.argsort(-row)
                sorted_args = sorted_args[np.where(row[sorted_args] > 0)]
                tmp = np.column_stack((feature_names[sorted_args], row[sorted_args]))
                tfidf_f.write("%s\n" % " ".join(map(":".join, tmp.tolist())))
