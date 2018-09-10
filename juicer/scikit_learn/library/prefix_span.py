# -*- coding: utf-8 -*-

counter = 0


class PrefixSpan:
    def __init__(self, db=[]):
        self.db = db
        self.genSdb()

    def genSdb(self):
        '''
        Generate mutual converting tables between db and sdb
        '''
        self.db2sdb = dict()
        self.sdb2db = list()
        count = 0
        self.sdb = list()
        for seq in self.db:
            newseq = list()
            for item in seq:
                if self.db2sdb.has_key(item):
                    pass
                else:
                    self.db2sdb[item] = count
                    self.sdb2db.append(item)
                    count += 1
                newseq.append(self.db2sdb[item])
            self.sdb.append(newseq)
        self.itemCount = count

    def run(self, min_sup=0.2, max_len=3):
        '''
        mine patterns with min_sup as the min support threshold
        '''
        self.min_sup = min_sup * len(self.db)
        L1Patterns = self.genL1Patterns()
        patterns = self.genPatterns(L1Patterns, max_len)
        self.sdbpatterns = L1Patterns + patterns

    def getPatterns(self):
        oriPatterns = list()
        for (pattern, sup, pdb) in self.sdbpatterns:
            oriPattern = list()
            for item in pattern:
                oriPattern.append(self.sdb2db[item])
            oriPatterns.append((oriPattern, sup))
        return oriPatterns

    def genL1Patterns(self):
        pattern = []
        sup = len(self.sdb)
        pdb = [(i, 0) for i in range(len(self.sdb))]
        L1Prefixes = self.span((pattern, sup, pdb))

        return L1Prefixes

    def genPatterns(self, prefixes, max_len):
        results = []
        for prefix in prefixes:
            if len(prefix[0]) < max_len:
                result = self.span(prefix)
                results += result

        if results != []:
            results += self.genPatterns(results, max_len)
        return results

    def span(self, prefix):
        (pattern, sup, pdb) = prefix

        itemSups = [0] * self.itemCount
        for (sid, pid) in pdb:
            itemAppear = [0] * self.itemCount
            for item in self.sdb[sid][pid:]:
                itemAppear[item] = 1
            itemSups = map(lambda x, y: x + y, itemSups, itemAppear)
        prefixes = list()
        for i in range(len(itemSups)):
            if itemSups[i] >= self.min_sup:
                newPattern = pattern + [i]
                newSup = itemSups[i]
                newPdb = list()
                for (sid, pid) in pdb:
                    for j in range(pid, len(self.sdb[sid])):
                        item = self.sdb[sid][j]
                        if item == i:
                            newPdb.append((sid, j + 1))
                            break
                prefixes.append((newPattern, newSup, newPdb))

        return prefixes