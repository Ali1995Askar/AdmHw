import csv
import math
import os
import random
from collections import Counter


class Node:
    def __init__(self, is_leaf=False, result=None, branches=None, specs=None):
        self.is_leaf = is_leaf
        self.result = result
        self.branches = branches
        self.specs = specs


def create_tree(rows, attributes, name_to_index, attr_type):
    gain, specs = get_best_split(rows, attributes, name_to_index, attr_type)

    if gain <= THRESHOLD:
        return Node(is_leaf=True,
                    result=get_common_class(rows))

    created_branches = create_branches(rows, specs)
    branches = {}
    for branch_name, branch_rows in created_branches.items():
        branches[branch_name] = create_tree(branch_rows, attributes, name_to_index, attr_type)

    return Node(is_leaf=False,
                branches=branches,
                specs=specs)


def predictId3(row, node):
    if node.is_leaf:
        return node.result

    if node.specs["is_continuous"]:
        if node.specs["values"][0] >= row[node.specs["attribute"]]:
            return predictId3(row, node.branches["true"])
        else:
            return predictId3(row, node.branches["false"])
    else:
        for value in node.specs["values"]:
            if value == row[node.specs["attribute"]]:
                return predictId3(row, node.branches[value])


def evaluation(rows, root):
    result = []
    for row in rows:
        result.append(predictId3(row, root) == row[TARGET])
    return result.count(True) / len(result) * 100


def print_tree(node, lvl=0, spacing=""):
    if node.is_leaf:
        print(spacing + "predictId3", node.result)
        return

    if node.specs["is_continuous"]:
        print(spacing + f'depth:{lvl}-child:{0} ---> Is {index_to_name[node.specs["attribute"]]} >= {node.specs["values"][0]}-> True":')
        print_tree(node.branches["true"], lvl + 1, spacing + "  ")

        print(spacing + f'depth:{lvl}-child:{1} ---> Is {index_to_name[node.specs["attribute"]]} >= {node.specs["values"][0]}-> False:')
        print_tree(node.branches["false"], lvl + 1, spacing + "  ")

    else:
        for child, value in enumerate(node.specs["values"]):
            # if condition.value in node.branches.keys():
            print(spacing + f'depth:{lvl}-child:{child} ---> Is {index_to_name[node.specs["attribute"]]} == {value}:')
            print_tree(node.branches[value], lvl + 1, spacing + "  ")


def create_branches(rows, specs):
    branches = {}
    partitioned = {"true": [], "false": []}

    # if the chosen attribute is continuous -> there is only one condition
    if specs["is_continuous"]:
        partitioned = divide_rows(rows,
                                specs["values"][0],
                                specs["attribute"],
                                specs["is_continuous"])
        return partitioned

    for value in specs["values"]:
        partitioned = divide_rows(rows,
                                value,
                                specs["attribute"],
                                specs["is_continuous"])

        branches[value] = partitioned["true"]
    return branches


def divide_rows(rows, value, attribute, is_contiuous):
    true, false = [], []
    splittingFunction = None
    # for int and float values
    if is_contiuous:
        splittingFunction = lambda row : row[attribute] >= value
    else: # for strings
        splittingFunction = lambda row : row[attribute] == value

    for row in rows:
        if splittingFunction(row):
            true.append(row)
        else:
            false.append(row)

    return {"true": true, "false": false}


def get_best_split(rows, attributes, name_to_index, attr_type):
    attrs_gains = {}
    cur_entropy = entropy(rows)

    for attr_name in attributes:
        attr = name_to_index[attr_name]
        if attr == TARGET:
            continue
        
        is_continuous = attr in attr_type["continuous"]
        values = get_distinct_values(rows, attr)
        processed_values = values if not is_continuous else [continuous_to_discrete(rows, values, attr, is_continuous)]

        classes, branches = {}, {}
        attr_specs = {
                        "is_continuous": is_continuous,
                        "attribute": attr,
                        "values": []
                    }
        for value in processed_values:
            branches = divide_rows(rows, value, attr, is_continuous)

            attr_specs["values"].append(value)

            if is_continuous == False:
                classes[value] = branches["true"]

        attr_branches = branches if is_continuous else classes
        attrs_gains[information_gain(attribute_entropy(attr_branches), cur_entropy)] = attr_specs
    attrs_gains = {key: val for (key, val) in sorted(attrs_gains.items(), reverse=True)}

    # returns the best info gain
    return next(iter(attrs_gains.items()))


def continuous_to_discrete(rows, values, attribute, is_continuous):
    best_entropy, best_value = 10000000, None
    for value in values:
        branches = divide_rows(rows,
                                value,
                                attribute,
                                is_continuous)

        # if this class doesn't split rows continue
        if branches["true"] == 0 or branches["false"] == 0:
            continue

        entropy_ = attribute_entropy(branches)
        if entropy_ < best_entropy:
            best_entropy = entropy_
            best_value = value

    return best_value


def entropy(rows):
    counts = classes_count(rows)
    tot = len(rows)
    result = 0
    for label in counts:
        prob_label = counts[label] / tot
        result += -prob_label * math.log2(prob_label)

    return result


def information_gain(branching_entropy, current_uncertainty):
    return current_uncertainty - branching_entropy


def attribute_entropy(branches):
    total = sum([len(val) for val in branches.values()])
    temp_sum = 0
    for val in branches.values():
        label_proportion = len(val) / total
        temp_sum += label_proportion * entropy(val)
    return temp_sum


def get_common_class(rows):
    classes = Counter([row[TARGET] for row in rows])
    return classes.most_common(1)[0][0]


def get_distinct_values(rows, attr):
    return set([row[attr] for row in rows])


def classes_count(rows):
    count = Counter([row[TARGET] for row in rows])
    return count


def read_csv(file_path, split=(100,0), shuffle=False, cast_numerics=True):
    with open(file_path, "r") as csv_file:
        dataset = csv.reader(csv_file)
        data_table = [row for row in dataset]
        headers = data_table[0]
        rows = data_table[1:]

        rows_num = len(rows)

        name_to_index = {name: index for index, name in enumerate(headers)}
        index_to_name = {index: name for index, name in enumerate(headers)}

        attr_type = {"categorical": set(),
                     "continuous": set()
                     }
        # cast numeric data to float
        # mark the attribute as continuous/categorical
        for attr_name in headers:

            c_idx = name_to_index[attr_name]
            is_continuous = False
            attr_values = []

            for r_idx in range(rows_num):
                val = rows[r_idx][c_idx]

                attr_values.append(val)

                numeric_type = val.isnumeric()
                is_continuous |= numeric_type
                if cast_numerics and numeric_type:
                    rows[r_idx][c_idx] = float(val)

            if is_continuous:
                attr_type["continuous"].add(c_idx)
            else:
                attr_type["categorical"].add(c_idx)

            # using "attr_values" to compute frequencies of each label of the attribute "attr_name"
            count = Counter(attr_values)
            fill_missing_data(rows, c_idx, count.most_common(1)[0])

        if shuffle:
            shuffle_data(rows)

        train_set, test_set = split_data(rows, rows_num, split)

        return {
            "headers": headers,
            "train_set": train_set,
            "test_set": test_set,
            "attr_type": attr_type,
            "name_to_index": name_to_index,
            "index_to_name": index_to_name
        }


def split_data(rows, size, split):

    train_ratio, _ = split

    split_idx = size * train_ratio // 100

    return rows[:split_idx], rows[split_idx:]


def shuffle_data(rows):
    random.seed(13.1)
    random.shuffle(rows)


def fill_missing_data(rows, attr, frequent_label):
    for i in range(len(rows)):
        if rows[i][attr] == "?":
            rows[i][attr] = frequent_label

module_dir = os.path.dirname(__file__)
file_path = os.path.join(module_dir, "heart_disease_male.csv")

data = read_csv(file_path, shuffle=True)
TARGET = len(data["headers"]) - 1
THRESHOLD = 0.0
index_to_name = data["index_to_name"]
root = create_tree(data["train_set"], data["headers"], data["name_to_index"], data["attr_type"])
