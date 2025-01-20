import numpy
import pandas as pd


def parse_data(filename):
    with open(filename, 'r') as fd:
        lines = fd.readlines()

    raw_data = pd.DataFrame({"lines":lines})

    raw_data["group_id"] = raw_data['lines'].str.startswith("#*").cumsum()

    raw_data['field'] = raw_data['lines'].str.extract(r"^(#\S)")
    raw_data = raw_data[raw_data['lines'] != "\n"]
    raw_data["value"] = raw_data['lines'].str.extract(r"^#\S(.*)").fillna("")
    raw_data.drop(["lines"], axis=1, inplace=True)
    aggregated_data = (raw_data.groupby(["group_id", "field"])["value"].apply(lambda x: x.tolist() if x.name == "#%" else " ".join(x)).reset_index())
    refined_data = aggregated_data.pivot(index="group_id", columns="field", values="value")
    refined_data = refined_data.rename(columns={
        "#*": "title",
        "#t": "year",
        "#c": "venue",
        "#@": "author",
        "#i": "index",
        "#!": "abstract",
        "#%": "citations"
        }
    )

    refined_data = refined_data.reset_index(drop=True)
    print(refined_data)

filepath = "/Users/ajeyk/neu_uml_workspace/test_data.txt"
parse_data(filepath)