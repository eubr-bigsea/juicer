import itertools
from pyspark.sql import Row
from pyspark.sql.functions import col


class LemonadeAssociativeRules(object):

    def __init__(self, itemsCol, freqCol, minConf=0.4, rulesCount=200):
        self.col_items = itemsCol
        self.col_freq = freqCol
        self.min_conf = minConf
        self.max_rules = rulesCount

    def run(self, df):
        itemsets = df.select(self.col_items, self.col_freq)\
                .rdd.map(lambda row: (tuple(set(row[0])), row[1])).cache()

        def gen_candidates(row):
            itemset, upper_support = row

            output = []
            for i in range(1, len(itemset)):
                for antecedent in itertools.combinations(itemset, r=i):
                    antecedent = tuple(set(antecedent))
                    consequent = tuple(set(set(itemset) - set(antecedent)))
                    output.append([antecedent, consequent, upper_support])

            return output

        rules = itemsets\
            .flatMap(gen_candidates)\
            .map(lambda item: (item[0], item)) \
            .join(itemsets) \
            .map(lambda item: (item[1][0][1],
                               (item[1][0][0], item[1][0][1],
                                item[1][0][2], item[1][1])))\
            .join(itemsets) \
            .map(lambda item: Row(antecedent=list(item[1][0][0]),
                                  consequent=list(item[1][0][1]),
                                  upper_support=item[1][0][2],
                                  lower_support=item[1][0][3],
                                  consequent_support=item[1][1]))\
            .toDF()

        rules = rules.\
            withColumn('confidence',
                       rules.upper_support/rules.lower_support)\
            .filter('confidence >= {}'.format(self.min_conf))\
            .orderBy(['confidence'], ascending=False)\
            .limit(self.max_rules)\
            .withColumn('lift',
                        col('confidence') / col('consequent_support'))\
            .withColumn('conviction',
                        (1 - col('consequent_support')) /
                        (1 - col('confidence')))\
            .withColumn('leverage',
                        col("upper_support") -
                        col("consequent_support") * col("lower_support"))\
            .withColumn('jaccard',
                        col("upper_support") /
                        (col("consequent_support") + col("lower_support")
                         - col("upper_support")))\
            .select("antecedent",
                    'consequent',
                    'confidence',
                    'lift',
                    'conviction',
                    'leverage',
                    'jaccard')

        itemsets.unpersist()

        return rules
