# coding=utf-8
from __future__ import unicode_literals

import base64
import itertools
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from html import escape  # python 3.x
except ImportError:
    from cgi import escape  # python 2.x


class BaseHtmlReport(object):
    pass


class HtmlImageReport(BaseHtmlReport):
    def __init__(self, image):
        self.image = image

    def generate(self):
        return base64.encodestring(self.image)


class SeabornChartReport(BaseHtmlReport):
    def __init__(self):
        pass

    # noinspection PyMethodMayBeStatic
    def jointplot(self, data, x, y):
        plt.style.use("seaborn-whitegrid")
        data_df = pd.DataFrame.from_records(data)
        sns.set(rc={'figure.figsize': (1, 1)})
        g = sns.jointplot(x=x, y=y, data=data_df)
        g.fig.subplots_adjust(top=.9, left=.15)
        fig_file = BytesIO()
        plt.savefig(fig_file, format='png', dpi=75)
        plt.close('all')
        return base64.b64encode(fig_file.getvalue())


class ConfusionMatrixImageReport(BaseHtmlReport):
    def __init__(self, cm, classes, normalize=False,
                 title='Confusion matrix', cmap=None,
                 axis=None):
        """
       This function prints and plots the confusion matrix.
       Normalization can be applied by setting `normalize=True`.
       """
        self.cm = cm
        self.classes = classes
        self.normalize = normalize
        self.title = title
        self.cmap = cmap
        if axis is not None:
            self.axis = axis
        else:
            self.axis = [_('Label'), _('Predicted')]

        if self.cmap is None:
            self.cmap = plt.cm.Blues

    def generate(self):
        if self.normalize:
            self.cm = self.cm.astype(
                'float') / self.cm.sum(axis=1)[:, np.newaxis]

        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        cax = ax1.imshow(self.cm, interpolation='nearest', cmap=self.cmap)
        fig.colorbar(cax)
        ax1.set_title(self.title)
        tick_marks = np.arange(len(self.classes))
        ax1.set_xticks(tick_marks)
        ax1.set_xticklabels(self.classes, rotation=45, fontsize=9)

        ax1.set_yticks(tick_marks)
        ax1.set_yticklabels(self.classes, fontsize=9)

        fmt = '.2f' if self.normalize else 'd'
        thresh = self.cm.max() / 2.
        for i, j in itertools.product(range(self.cm.shape[0]),
                                      range(self.cm.shape[1])):
            ax1.text(j, i, format(int(self.cm[i, j]), fmt),
                     horizontalalignment="center",
                     color="white" if self.cm[i, j] > thresh else "black")

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        ax1.set_ylabel(self.axis[0])
        ax1.set_xlabel(self.axis[1])
        fig_file = BytesIO()
        fig.savefig(fig_file, format='png')

        plt.close(fig)
        plt.close('all')

        return base64.b64encode(fig_file.getvalue())


class SimpleTableReport(BaseHtmlReport):
    def __init__(self, table_class, headers, rows, title=None, numbered=False):
        self.table_class = table_class
        self.headers = headers
        self.rows = rows
        self.title = title
        self.numbered = numbered

    def generate(self):
        code = []
        if self.title:
            code.append('<h4>{}</h4>'.format(self.title))
        code.append('<table class="{}"><thead><tr>'.format(self.table_class))
        if self.numbered:
            code.append('<th>#</th>')

        for col in self.headers:
            code.append(u'<th>{}</th>'.format(escape(unicode(col))))
        code.append('</tr></thead>')

        code.append('<tbody>')
        for i, row in enumerate(self.rows):
            code.append('<tr>')
            if self.numbered:
                code.append('<td>{}</td>'.format(i + 1))
            for col in row:
                if isinstance(col, str):
                    col = col.decode('utf8')
                code.append(u'<td>{}</td>'.format(col))
            code.append('</tr>')

        code.append('</tbody>')
        code.append('</table>')
        return ''.join(code)
