# from tools.csv_tools import describe_csv

# print(describe_csv("students.csv"))

from tools.csv_tools import filter_csv

print(filter_csv(
    "students.csv",
    [
        ("score", ">", 70),
        ("attendance", ">", 80)
    ],
    "and"
))