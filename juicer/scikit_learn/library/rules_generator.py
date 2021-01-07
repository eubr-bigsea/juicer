# coding=utf-8
import itertools
import pandas as pd


class RulesGenerator:

    def __init__(self, min_conf, max_len):
        self.min_conf = min_conf
        self.max_len = max_len

    def get_rules(self, freq_items, col_item, col_freq):
        """
        Given a set of frequent itemsets, return a list
        of association rules in the form
        [antecedent, consequent, confidence, lift, conviction,
        leverage, jaccard]
        """
        patterns = {tuple(k): v for (k, v) in
                    freq_items[[col_item, col_freq]].to_numpy().tolist()}
        rules = []

        for itemset in patterns.keys():
            upper_support = patterns[itemset]

            for i in range(len(itemset) - 1, 0, -1):

                for antecedent in itertools.combinations(itemset, r=i):
                    antecedent = tuple(sorted(antecedent))
                    consequent = tuple(sorted(set(itemset) - set(antecedent)))

                    if antecedent in patterns:

                        lower_support = patterns[antecedent]
                        confidence = upper_support / lower_support

                        consequent_support = patterns.get(consequent, 0)

                        # lift
                        lift = float("inf")
                        if consequent_support != 0:
                            lift = confidence / consequent_support

                        # conviction
                        conviction = float("inf")
                        if (1 - confidence) != 0:
                            conviction = (1 - consequent_support) / (
                                        1 - confidence)

                        # leverage
                        leverage = upper_support - consequent_support * \
                                   lower_support

                        # jaccard
                        try:
                            jaccard = upper_support / (
                                    consequent_support + lower_support -
                                    upper_support)
                        except ZeroDivisionError:
                            jaccard = 1.0

                        rules.append(
                                [list(antecedent), list(consequent), confidence,
                                 lift, conviction, leverage, jaccard])

        columns = ['Pre-Rule', 'Post-Rule', 'Confidence', 'Lift', 'Conviction',
                   "Leverage", 'Jaccard']
        rules = pd.DataFrame(rules, columns=columns)

        if self.min_conf > 0:
            rules = rules[rules['Confidence'] > self.min_conf]

        rules.sort_values(by='Confidence', inplace=True, ascending=False)
        if 0 < self.max_len < len(rules):
            rules = rules.head(self.max_len)

        return rules
