import csv
import random

main_filename = '/home/pulkit/research/sexismclassification/data/data_new.csv'   
out_filename = '/home/pulkit/research/sexismclassification/data/samples_suppl.txt'   
latex_filename = '/home/pulkit/research/sexismclassification/data/latex_samples_suppl.txt'   

num_entries = {
  'role stereotyping': 10,
  'attribute stereotyping': 10,
  'hyper-sexualization': 10,
  'internalized sexism': 10,
  'hostile work environment': 10, 
  'pay gap': 10,
  'body shaming': 10,
  'denial or trivialisation': 10, 
  'threats': 10, 
  'sexual assault': 10,
  'rape': 10,
  'sexual harassment': 10,
  'moral policing': 10,
  'tone policing': 10,
  'victim blaming': 10,
  'slut shaming': 10,
  'mom shaming': 10,
  'period shaming': 10,
  'other': 10,
  'religion-based sexism': 10,
  'mansplaining': 10,
  'gaslighting': 10,
  'physical violence': 10,
  # '': 13
  }

cat_dict = {}
cnt = 0
with open(main_filename, 'r') as csvfile:
  reader = csv.DictReader(csvfile, delimiter = '\t')
  for entry in reader:
      var = entry["category_2"]
      var = var.lower().split(',')

      for v in var:
        if v in cat_dict:
          cat_dict[v].append(entry)
        else:
          cat_dict[v]= [entry]

      cnt += 1

print("total entries: %d" % cnt)

rows = []
for cat, record_list in cat_dict.items():
  s_list = random.sample(record_list, min(len(record_list), num_entries[cat]))
  for t in s_list:
    row = {}
    row['Sexism instance'] = t["post"].strip()
    row['Categories'] = t["category_2"].strip()
    rows.append(row)

random.shuffle(rows)

with open(out_filename, 'w') as fO:
  writer_m = csv.DictWriter(fO, fieldnames = ['Sexism instance', 'Categories'], delimiter = '\t')
  writer_m.writeheader()
  for r in rows:
    writer_m.writerow(r)

with open(latex_filename, 'w') as f:
  f.write("Sexism instance\tCategories\n")
  # f.write("\\hline\n")
  for r in rows:
    f.write("%s\t%s\n" % (r['Sexism instance'], r['Categories']))
    # f.write("\\hline\n")
