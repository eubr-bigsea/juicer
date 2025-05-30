# coding=utf-8


import base64
import datetime
import itertools
import os
import re
from io import BytesIO
from plotly.subplots import make_subplots
from collections.abc import Iterable
from gettext import gettext
import plotly.graph_objects as go
import jinja2
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


class DummyLock(object):
    def __enter__(self):
        return True

    def __exit__(self, _type, value, traceback):
        return True


class HtmlImageReport(BaseHtmlReport):
    def __init__(self, image):
        self.image = image

    def generate(self):
        return base64.encodebytes(self.image)


class PlotlyChartReport(BaseHtmlReport):
    @staticmethod
    def plot_jointplot(x, y,
                  width=800,
                  height=800,
                  n_bins=30,
                  point_size=1,
                  point_color='blue',
                  point_opacity=0.6,
                  hist_color='blue',
                  hist_opacity=0.7,
                  x_label='X',
                  y_label='Y',
                  title=None):
        """
        Create a joint plot showing scatter plot with marginal distributions.

        Parameters:
        -----------
        x : array-like
            Data for x-axis
        y : array-like
            Data for y-axis
        width : int
            Figure width in pixels
        height : int
            Figure height in pixels
        n_bins : int
            Number of bins for histograms
        point_size : int
            Size of scatter plot points
        point_color : str
            Color of scatter plot points
        point_opacity : float
            Opacity of scatter plot points (0 to 1)
        hist_color : str
            Color of histogram bars
        hist_opacity : float
            Opacity of histogram bars (0 to 1)
        x_label : str
            Label for x-axis
        y_label : str
            Label for y-axis
        title : str
            Plot title (optional)
        """

        fig = make_subplots(
            rows=2, cols=2,
            column_widths=[0.7, 0.3],
            row_heights=[0.3, 0.7],
            vertical_spacing=0.03,
            horizontal_spacing=0.03
        )

        # Add scatter plot
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode='markers',
                marker=dict(
                    size=point_size,
                    color=point_color,
                    opacity=point_opacity
                ),
                showlegend=False
            ),
            row=2, col=1
        )

        # Add x-axis histogram
        fig.add_trace(
            go.Histogram(
                x=x,
                nbinsx=n_bins,
                marker_color=hist_color,
                opacity=hist_opacity,
                showlegend=False
            ),
            row=1, col=1
        )

        # Add y-axis histogram
        fig.add_trace(
            go.Histogram(
                y=y,
                nbinsy=n_bins,
                marker_color=hist_color,
                opacity=hist_opacity,
                showlegend=False
            ),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            title=title,
            width=width,
            height=height,
            template='plotly_white',
            showlegend=False,
            xaxis2=dict(
                showgrid=True,
                zeroline=False,
                title=x_label
            ),
            yaxis2=dict(
                showgrid=True,
                zeroline=False,
                title=y_label
            ),
            xaxis1=dict(
                showticklabels=False,
                showgrid=False
            ),
            yaxis1=dict(
                showticklabels=False,
                showgrid=False
            ),
            xaxis3=dict(
                showticklabels=False,
                showgrid=False
            ),
            yaxis3=dict(
                showticklabels=False,
                showgrid=False
            ),
            bargap=0.1,
            margin=dict(
                l=50,
                r=50,
                b=50,
                t=50 if title is None else 70
            )
        )

        return fig.to_json()

    @staticmethod
    def plot(title, x_label, y_label, *args, **kwargs):
        if "submission_lock" in kwargs:
            submission_lock = kwargs.get("submission_lock")
            del kwargs["submission_lock"]
        else:
            submission_lock = DummyLock()

        show_legend = bool(kwargs.pop("show_legend", False))
        trace_names = kwargs.pop("traces", [])
        mode = kwargs.pop("mode", ["lines"])

        with submission_lock:
            fig = go.Figure()
            fig.update_layout(
                title=title,
                xaxis_title=x_label,
                yaxis_title=y_label,
                template="plotly_white",
                autosize=True,  # Enable autosize
            )

            for i, arg in enumerate(args):
                fig.add_trace(
                    go.Scatter(
                        x=arg[0], y=arg[1], name=trace_names[i],
                        mode=mode[i],
                        marker=dict(
                            size=3,
                            symbol='circle'
                    )
                    )
                )

            if show_legend:
                fig.update_layout(legend_title_text=gettext("Legend"))

            return fig.to_json()


class MatplotlibChartReport(BaseHtmlReport):
    @staticmethod
    def plot(title, x_label, y_label, *args, **kwargs):
        if "submission_lock" in kwargs:
            submission_lock = kwargs.get("submission_lock")
            del kwargs["submission_lock"]
        else:
            submission_lock = DummyLock()

        show_legend = bool(kwargs.pop("show_legend", False))

        with submission_lock:
            plt.clf()
            # plt.rcParams["figure.figsize"] = (6, 6)
            plt.figure(figsize=(10, 6), dpi=100)
            plt.title(title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.style.use("seaborn-whitegrid")
            if show_legend:
                plt.legend([x_label, y_label], loc="best", fancybox=True)
            plt.plot(*args, **kwargs)
            fig_file = BytesIO()
            plt.savefig(fig_file, format="png", dpi=75)
            plt.close("all")
            return base64.b64encode(fig_file.getvalue()).decode("utf-8")


class SeabornChartReport(BaseHtmlReport):
    @staticmethod
    def jointplot(
        data, x, y, title, x_label, y_label, submission_lock=DummyLock()
    ):
        with submission_lock:
            plt.clf()
            plt.style.use("seaborn-whitegrid")
            plt.title(title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            data_df = pd.DataFrame.from_records(data)
            sns.set(rc={"figure.figsize": (1, 1)})
            g = sns.jointplot(x=x, y=y, data=data_df)
            g.set_axis_labels(x_label, y_label)

            g.fig.subplots_adjust(top=0.9, left=0.15)
            fig_file = BytesIO()
            plt.axhline(y=0, color="red", linestyle="--")
            plt.savefig(fig_file, format="png", dpi=75)
            plt.close("all")
            return base64.b64encode(fig_file.getvalue()).decode("utf-8")


class ConfusionMatrixImageReport(BaseHtmlReport):
    def __init__(
        self, cm, classes, normalize=False, title=None, cmap=None, axis=None
    ):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if title is None:
            title = gettext("Confusion matrix")
        self.cm = cm
        self.classes = classes
        self.normalize = normalize
        self.title = title
        self.cmap = cmap
        if axis is not None:
            self.axis = axis
        else:
            self.axis = [gettext("Label"), gettext("Predicted")]

    def generate(self, submission_lock=DummyLock()):
        with submission_lock:
            if self.cmap is None:
                self.cmap = plt.cm.Blues

            if self.normalize:
                self.cm = (
                    self.cm.astype("float") / self.cm.sum(axis=1)[:, np.newaxis]
                )
            plt.clf()
            plt.rcParams["figure.figsize"] = (6, 6)
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.grid(False)
            cax = ax1.imshow(self.cm, interpolation="nearest", cmap=self.cmap)
            fig.colorbar(cax)
            ax1.set_title(self.title)
            tick_marks = np.arange(len(self.classes))
            ax1.set_xticks(tick_marks)
            ax1.set_xticklabels(self.classes, rotation=45, fontsize=9)

            ax1.set_yticks(tick_marks)
            ax1.set_yticklabels(self.classes, fontsize=9)

            fmt = ".2f" if self.normalize else "d"
            thresh = self.cm.max() / 2.0
            for i, j in itertools.product(
                iter(list(range(self.cm.shape[0]))),
                iter(list(range(self.cm.shape[1]))),
            ):
                ax1.text(
                    j,
                    i,
                    format(int(self.cm[i, j]), fmt),
                    horizontalalignment="center",
                    color="white" if self.cm[i, j] > thresh else "black",
                )

            # issue: https://github.com/matplotlib/matplotlib/issues/19608
            fig.tight_layout(rect=[0.05, 0.03, 1, 0.95])

            ax1.set_ylabel(self.axis[0])
            ax1.set_xlabel(self.axis[1])
            fig_file = BytesIO()
            fig.savefig(fig_file, format="png")

            plt.close(fig)
            plt.close("all")

            return base64.b64encode(fig_file.getvalue()).decode("utf-8")


class SimpleTableReport(BaseHtmlReport):
    def __init__(self, table_class, headers, rows, title=None, numbered=None):
        self.table_class = table_class
        self.headers = headers
        self.rows = rows
        self.title = title
        self.numbered = numbered

    def generate(self):
        code = []
        if self.title:
            code.append("<h6>{}</h6>".format(self.title))
        code.append('<table class="{}">'.format(self.table_class))
        if self.headers:
            code.append("<thead><tr>")
            if self.numbered is not None:
                code.append("<th>#</th>")

            for col in self.headers:
                code.append("<th>{}</th>".format(escape(col)))
            # code.append('<th></th>')
            code.append("</tr></thead>")

        code.append("<tbody>")
        for i, row in enumerate(self.rows):
            code.append("<tr>")
            if self.numbered is not None:
                code.append("<td>{}</td>".format(i + self.numbered))
            for col in row:
                if col.__class__.__name__ == "DenseMatrix":
                    new_rows = col.toArray().tolist()
                    report = SimpleTableReport(self.table_class, [], new_rows)
                    code.append("<td>{}</td>".format(report.generate()))
                elif col.__class__.__name__ in ["DenseVector", "SparseVector"]:
                    vector = list(col.toArray())
                    vector_as_str = ", ".join([str(c) for c in vector])
                    code.append("<td>{}</td>".format(vector_as_str))
                elif isinstance(col, Iterable) and not isinstance(col, str):
                    code.append(
                        "<td>{}</td>".format(", ".join([str(c) for c in col]))
                    )
                else:
                    code.append("<td>{}</td>".format(col))
            # code.append('<td>{}</td>'.format(col.__class__.__name__))
            code.append("</tr>")

        code.append("</tbody>")
        code.append("</table>")
        return "".join(code)


class FairnessBiasReport(BaseHtmlReport):
    explanations = {
        "pred_pos_ratio_k_parity": {
            "title": "Equal Parity",
            "description": """This criteria considers an attribute to have equal parity is
                every group is equally represented in the selected set.
                For example, if race (with possible values of white, black,
                other) has equal parity, it implies that all three races are
                equally represented (33% each)in the selected/intervention
                set. """,
            "usage": """If your desired outcome is to intervene equally on people
                from all races, then you care about this criteria. """,
        },
        "pred_pos_ratio_g_parity": {
            "title": "Proportional Parity",
            "description": """This criteria considers an attribute to have proportional
                parity if every group is represented proportionally to their
                share of the population. For example, if race with possible
                values of white, black, other being 50%, 30%, 20% of the
                population respectively) has proportional parity, it implies
                that all three races are represented in the same proportions
                (50%, 30%, 20%) in the selected set. """,
            "usage": """If your desired outcome is to intervene proportionally
            on people from all races, then you care about this criteria.""",
        },
        "fpr_parity": {
            "title": "False Positive Rate Parity",
            "description": """This criteria considers an attribute to have
            False Positive parity if every group has the same False Positive
            Error Rate. For example, if race has false positive parity, it
            implies that all three races have the same False Positive Error
            Rate. """,
            "usage": """If your desired outcome is to intervene proportionally
            on people from all races, then you care about this criteria. """,
        },
        "fdr_parity": {
            "title": "False Discovery Rate Parity",
            "description": """This criteria considers an attribute to have False
             Discovery Rate parity if every group has the same False Discovery
             Error Rate. For example, if race has false discovery parity, it
             implies that all three races have the same False Discvery Error
             Rate. """,
            "usage": """If your desired outcome is to make false
             positive errors equally on people from all races, then you care
             about this criteria. This is important in cases where your
             intervention is punitive and can hurt individuals and where
             you are selecting a very small group for interventions. """,
        },
        "fnr_parity": {
            "title": "False Positive Rate Parity",
            "description": """This criteria considers an attribute to have False
             Positive parity if every group has the same False Positive Error
             Rate. For example, if race has false positive parity, it implies
             that all three races have the same False Positive Error Rate. """,
            "usage": """If your desired outcome is to make false positive errors
             equally on people from all races, then you care about this
             criteria. This is important in cases where your intervention is
             punitive and has a risk of adverse outcomes for individuals.
             Using this criteria allows you to make sure that you are not
             making false positive mistakes about any single group
             disproportionately. """,
        },
        "for_parity": {
            "title": "False Omission Rate Parity",
            "description": """This criteria considers an attribute to have False
             Omission Rate parity if every group has the same False Omission
             Error Rate. For example, if race has false omission parity, it
             implies that all three races have the same False Omission
             Error Rate.""",
            "usage": """If your desired outcome is to make false negative errors
             equally on people from all races, then you care about this
             criteria. This is important in cases where your intervention is
             assistive (providing help social services for example) and missing
              an individual could lead to adverse outcomes for them , and where
               you are selecting a very small group for interventions. Using
               this criteria allows you to make sure that you're not missing
               people from certain groups disproportionately.""",
        },
    }

    def __init__(self, df, sensitive, baseline_value):
        self.df = df
        self.sensitive = sensitive
        self.baseline_value = baseline_value

    def generate(self):
        data = [row1.asDict() for row1 in self.df.collect()]
        order = [
            "pred_pos_ratio_k_parity",
            "pred_pos_ratio_g_parity",
            "fpr_parity",
            "fdr_parity",
            "fnr_parity",
            "for_parity",
        ]
        summary = [
            [
                v,
                all([row[v] for row in data]),
                [
                    [
                        row[self.sensitive],
                        row[v],
                        row[v.replace("parity", "disparity")],
                    ]
                    for row in data
                ],
            ]
            for v in order
        ]

        template_loader = jinja2.FileSystemLoader(
            searchpath=os.path.dirname(__file__)
        )

        template_env = jinja2.Environment(loader=template_loader)
        template = template_env.get_template("templates/bias-report.html")

        ctx = {
            "date": datetime.datetime.now().isoformat(),
            "_": gettext,
            "data": data,
            "tau": 0.8,
            "reference": self.baseline_value,
            "summary": summary,
            "explanations": self.explanations,
            "attributes": ", ".join([self.sensitive]),
        }
        return template.render(ctx)


class AreaUnderCurveReport(BaseHtmlReport):
    def __init__(self, x_val, y_val, title=None, curve_type="roc"):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if title is None:
            title = gettext("Area under curve")
        self.title = title
        self.x_val = x_val
        self.y_val = y_val
        self.curve_type = curve_type

    def generate(self, submission_lock):
        with submission_lock:
            plt.style.use("seaborn-whitegrid")
            plt.figure()
            plt.plot(self.x_val, self.y_val)
            if self.curve_type == "roc":
                plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
                plt.xlabel(gettext("False positive rate"))
                plt.ylabel(gettext("True positive rate"))
            else:
                plt.xlabel(gettext("Precision"))
                plt.ylabel(gettext("Recall"))

            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])

            plt.title(self.title)
            plt.legend(loc="lower right")

            fig_file = BytesIO()
            plt.savefig(fig_file, format="png", dpi=75)
            plt.close("all")
            return base64.b64encode(fig_file.getvalue()).decode("utf-8")


class DecisionTreeReport(BaseHtmlReport):
    """
    Report for decision trees from Spark.
    Based on: https://github.com/tristaneljed/Decision-Tree-Visualization-Spark/
    """

    def __init__(self, model, spark_schema, features_col, features):
        self.model = model
        self.spark_schema = spark_schema
        self.features_col = features_col
        self.tree = model.toDebugString
        self.features = features

    def _tree_json(self):
        data = []
        for line in self.tree.splitlines():
            if line.strip():
                line = line.strip()
                data.append(line)
        return {"name": "Root", "children": self._parse(data[1:])}

    def _parse(self, lines):
        block = []
        import pdb

        pdb.set_trace()
        while lines:
            if lines[0].startswith("If"):
                bl = (
                    " ".join(lines.pop(0).split()[1:])
                    .replace("(", "")
                    .replace(")", "")
                )
                block.append({"name": bl, "children": self._parse(lines)})

                if lines[0].startswith("Else"):
                    be = (
                        " ".join(lines.pop(0).split()[1:])
                        .replace("(", "")
                        .replace(")", "")
                    )
                    block.append({"name": be, "children": self._parse(lines)})
            elif not lines[0].startswith(("If", "Else")):
                block.append({"name": lines.pop(0)})
            else:
                break
        return block

    def generate(self):
        from juicer.spark.util.tree_visualization import get_graph_from_model

        result = "<h6>{}</h6>{}".format(
            gettext("Tree"),
            get_graph_from_model(
                self.model, self.spark_schema, self.features
            ).decode("utf-8"),
        )
        mapping = dict(
            [(f"feature {x}", v) for (x, v) in enumerate(self.features)]
        )
        pattern = re.compile("|".join(mapping.keys()), re.IGNORECASE)
        return pattern.sub(lambda m: mapping[m.group(0).lower()], result)
