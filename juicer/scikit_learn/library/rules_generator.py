
from itertools import chain, combinations
import pandas as pd


class RulesGenerator:

    def filter_rules(self, rules, count, max_rules, pos):
        total, partial = count
        if total > max_rules:
            gets = 0
            for i in range(pos):
                gets += partial[i]
            number = max_rules - gets
            if number > partial[pos]:
                number = partial[pos]
            if number < 0:
                number = 0
            return rules.head(number)
        else:
            return rules

    def get_rules(self, freq_items, col_item, col_freq,  min_confidence):
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
                        if confidence > min_confidence:
                            r = [element, remain, confidence]
                            list_rules.append(r)

        cols = ['Pre-Rule', 'Post-Rule', 'confidence']
        rules = pd.DataFrame(list_rules, columns=cols)

        return rules

    def subsets(self, arr):
        return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])

    def get_support(self, element, freqset, col_item, col_freq):

        for t, s in zip(freqset[col_item].values, freqset[col_freq].values):
            if element == list(t):
                return s
        return float("inf")