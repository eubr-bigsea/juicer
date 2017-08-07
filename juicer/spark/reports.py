import base64

try:
    from html import escape  # python 3.x
except ImportError:
    from cgi import escape  # python 2.x

print(escape("<"))


class BaseHtmlReport(object):
    pass


class HtmlImageReport(BaseHtmlReport):
    def __init__(self, image):
        self.image = image

    def generate(self):
        return base64.encodestring(self.image)


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
            code.append('<th>{}</th>'.format(escape(str(col))))
        code.append('</tr></thead>')

        code.append('<tbody>')
        for i, row in enumerate(self.rows):
            code.append('<tr>')
            if self.numbered:
                code.append('<td>{}</td>'.format(i + 1))
            for col in row:
                code.append('<td>{}</td>'.format(escape(str(col))))
            code.append('</tr>')

        code.append('</tbody>')
        code.append('</table>')
        return ''.join(code)
