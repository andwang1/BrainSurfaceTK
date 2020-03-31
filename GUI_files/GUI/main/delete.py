import csv
with open("/home/cemlyn/Documents/Projects/MScGroupProject/data/meta_data.tsv") as foo:
    reader = csv.reader(foo, delimiter='\t')
    for row in reader:
        print(type(row))
