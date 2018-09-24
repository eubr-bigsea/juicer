# -*- coding: utf-8 -*-

"""
PrefixSpan.

Adapted implementation of github:
https://github.com/manohar-ch/prefixspan/tree/master
"""


class PrefixSpan:

    def __init__(self, db=None):
        if db is None:
            db = []
        self.db = db

        self.min_sup = 0
        self.item_count = 0
        self.sdb = list()
        self.db2sdb = dict()
        self.sdb2db = list()
        self.sdb_patterns = list()

        self.gen_sdb()

    def gen_sdb(self):
        """
        Generate mutual converting tables between db and sdb
        """

        count = 0
        for seq in self.db:
            newseq = list()
            for item in seq:
                if item in self.db2sdb:
                    pass
                else:
                    self.db2sdb[item] = count
                    self.sdb2db.append(item)
                    count += 1
                newseq.append(self.db2sdb[item])
            self.sdb.append(newseq)
        self.item_count = count

    def run(self, min_sup=0.2, max_len=3):
        """
        mine patterns with min_sup as the min support threshold
        """
        self.min_sup = int(min_sup * len(self.db))
        l1_patterns = self.gen_l1_patterns()
        patterns = self.gen_patterns(l1_patterns, max_len)
        self.sdb_patterns = l1_patterns + patterns

    def get_patterns(self):
        ori_patterns = list()
        for (pattern, sup, pdb) in self.sdb_patterns:
            ori_pattern = list()
            for item in pattern:
                ori_pattern.append(self.sdb2db[item])
            ori_patterns.append((ori_pattern, sup))
        return ori_patterns

    def gen_l1_patterns(self):
        pattern = []
        sup = len(self.sdb)
        pdb = [(i, 0) for i in range(len(self.sdb))]
        l1_prefixes = self.span((pattern, sup, pdb))

        return l1_prefixes

    def gen_patterns(self, prefixes, max_len):
        results = []
        for prefix in prefixes:
            if len(prefix[0]) < max_len:
                result = self.span(prefix)
                results += result

        if len(results) > 0:
            results += self.gen_patterns(results, max_len)
        return results

    def span(self, prefix):
        (pattern, sup, pdb) = prefix

        item_sups = [0] * self.item_count
        for (sid, pid) in pdb:
            item_appear = [0] * self.item_count
            for item in self.sdb[sid][pid:]:
                item_appear[item] = 1
            item_sups = map(lambda x, y: x + y, item_sups, item_appear)

        prefixes = list()
        for i in range(len(item_sups)):
            if item_sups[i] >= self.min_sup:
                new_pattern = pattern + [i]
                new_sup = item_sups[i]
                new_pdb = list()
                for (sid, pid) in pdb:
                    for j in range(pid, len(self.sdb[sid])):
                        item = self.sdb[sid][j]
                        if item == i:
                            new_pdb.append((sid, j + 1))
                            break
                prefixes.append((new_pattern, new_sup, new_pdb))

        return prefixes
