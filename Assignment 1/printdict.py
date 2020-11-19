import sys
import json
import random


def load_posting_lists():
    global inv_index, file
    with open(file) as f:
        inv_index = json.load(f)


def print_formatted():
    format_string = "{token}:{df}:{offset}"
    for token, docs in inv_index.items():
        print(format_string.format(token=token, df=len(docs), offset=random.randint(1, 33554432)))


if __name__ == "__main__":
    file = sys.argv[1][:sys.argv[1].rfind('.')] + ".idx"
    inv_index = {}
    load_posting_lists()
    print_formatted()
