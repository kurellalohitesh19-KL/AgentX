"""Tool registry. To add a tool: subclass Tool in a new file, then add it to build_tools()."""


class Tool:
    name = "tool"
    description = ""
    args_doc = "{}"

    def execute(self, **kwargs):
        raise NotImplementedError


def build_tools():
    from tools.calculator import Calculator
    from tools.data_query import ListFiles, ReadFile, SqlQuery
    from tools.web_search import WebSearch
    tools = [ListFiles(), ReadFile(), SqlQuery(), Calculator(), WebSearch()]
    return {t.name: t for t in tools}
