
from itertools import chain, combinations
import pandas as pd


class RulesGenerator:

    def __init__(self, min_conf, max_len):
        self.min_conf = min_conf
        self.max_len = max_len

    def get_rules(self, freq_items, col_item, col_freq):
        list_rules = []
        for index, row in freq_items.iterrows():
            item = row[col_item]
            support = row[col_freq]
            if len(item) > 0:
                sub_sets = [list(x) for x in self.subsets(item)]

                for element in sub_sets:
                    remain = list(set(item).difference(element))

                    if len(remain) > 0:
                        num = float(support)
                        den = self.get_support(element, freq_items,
                                               col_item, col_freq)
                        confidence = num/den
                        if confidence > self.min_conf:
                            r = [element, remain, confidence]
                            list_rules.append(r)

        cols = ['Pre-Rule', 'Post-Rule', 'confidence']
        rules = pd.DataFrame(list_rules, columns=cols)

        rules.sort_values(by='confidence', inplace=True, ascending=False)
        if self.max_len != -1 and self.max_len < len(rules):
            rules = rules.head(self.max_len)

        return rules

    def subsets(self, arr):
        return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])

    def get_support(self, element, freqset, col_item, col_freq):
        for t, s in zip(freqset[col_item].values, freqset[col_freq].values):
            if element == list(t):
                return s
        return float("inf")
