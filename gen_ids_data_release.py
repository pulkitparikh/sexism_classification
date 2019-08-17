import csv
import re
import json

data_file_name = '/home/pulkit/research/extented_sexismclassification/data/data_new.csv'
p1_path = '/home/pulkit/research/detectSexism/from_nisarg_server/sexism-research/human-annotations/'
out_file_name = '/home/pulkit/research/extented_sexismclassification/data/ids_labels.tsv'

re1 = re.compile(r'(<\s*h2\s*class\s*=\s*"entry-title"\s*><a.+title\s*=\s*")(.*)("\s*>)(.*)(</a></h2>)')

max_selections = 10

ORG_FOR_LMAP = {
  0: 'Role stereotyping',
  1: 'Attribute stereotyping',
  2: 'Hyper-sexualization (excluding body shaming)',
  3: 'Internalized sexism',
  4: 'Hostile work environment (excluding pay gap)', 
  5: 'Body shaming',
  6: 'Denial or trivialisation of sexist misconduct', 
  7: 'Threats', 
  8: 'Sexual assault (excluding rape)',
  9: 'Sexual harassment (excluding physical contact)',
  10: 'Moral policing (excluding tone policing)',
  11: 'Slut shaming',
  12: 'Motherhood-related discrimination',
  13: 'Other',
  14: 'Pay gap',
  15: 'Rape',
  16: 'Tone policing',
  17: 'Victim blaming',
  18: 'Menstruation-related discrimination',
  19: 'Religion-based sexism',
  20: 'Mansplaining',
  21: 'Gaslighting',
  22: 'Physical violence (excluding sexual violence)',
  # 13: 'non-sexist'  
}

ORG_LABEL_MAP = {
  'role stereotyping': 0,
  'attribute stereotyping': 1,
  'hyper-sexualization': 2,
  'internalized sexism': 3,
  'hostile work environment': 4, 
  'pay gap': 14,
  'body shaming': 5,
  'denial or trivialisation': 6, 
  'threats': 7, 
  'sexual assault': 8,
  'rape': 15,
  'sexual harassment': 9,
  'moral policing': 10,
  'tone policing': 16,
  'victim blaming': 17,
  'slut shaming': 11,
  'mom shaming': 12,
  'period shaming': 18,
  'other': 13,
  'religion-based sexism': 19,
  'mansplaining': 20,
  'gaslighting': 21,
  'physical violence': 22,
  # '': 13
  }

reg_st = []
for i in range(max_selections):
	reg_st.append(re.compile(r'<span class="highlight annotate." annotateid="%d" style="text-shadow.*?">' % (i+1)))
reg_gen_st = re.compile(r'<span class="highlight annotate." annotateid="." style="text-shadow.*?">')
reg_end = re.compile(r'</span>')
reg_br_st = re.compile(r'^\s*?<br>')
reg_br = re.compile(r'<br>')

with open(data_file_name,'r') as f:
	reader = csv.DictReader(f, delimiter = '\t')
	rows = list(reader)
with open(out_file_name, 'w') as f_fin:
	w_fin = csv.DictWriter(f_fin, fieldnames = ['id', 'labels'], delimiter = '\t')
	w_fin.writeheader()

	for ind, row in enumerate(rows):
		parts = row['filename'].split('__')
		att_num = parts[1]
		assert(att_num == row['no_Annotation'])
		f_path = p1_path + parts[0] + '.txt'
		with open(f_path) as g:
			f_data = json.load(g)
		# post = str(f_data["metadata"]["html"].encode("utf-8"))
		post = f_data["metadata"]["html"]

		att_num_int = int(att_num)
		end_indice_data = [(m.start(), 'end', m.group()) for m in reg_end.finditer(post)]
		# print(att_num_int)
		# print(end_indices)
		esp_id = ('%s_' % f_data["metadata"]["post_id"])
		post_recon = ''
		match_data = [(m.start(), 'start', m.group()) for m in reg_st[att_num_int-1].finditer(post)]

		all_id_data = [(m.start(), 'start', m.group()) for m in reg_gen_st.finditer(post)]
		br_data = sorted([(m.start(), 'end', m.group()) for m in reg_br.finditer(post)], key=lambda x: x[0])
		comb_all_data = sorted(all_id_data + end_indice_data + br_data[1:], key=lambda x: x[0])
		prev = reg_br_st.search(post).end()
		char_rem_dict = {}
		for ind, entry in enumerate(comb_all_data):
			char_rem_dict[entry[0]] = entry[0] - prev
			prev += len(entry[2])
		# print(char_rem_dict)	
		comb_data = sorted(match_data + end_indice_data, key=lambda x: x[0])

		cur_count = 0
		prev_0 = -1
		# print(comb_data, "\n")
		for ind, entry in enumerate(comb_data):
			if entry[1] == 'start':
				cur_count += 1
			else:
				cur_count -= 1
			if cur_count < 0:
				cur_count = 0
				prev_0 = ind
			elif cur_count == 0:
				start_ind = comb_data[prev_0+1][0] + len(comb_data[prev_0+1][2])
				cur_post_recon = post[start_ind:entry[0]]
				for i in range(prev_0+2, ind):
					cur_post_recon = cur_post_recon.replace(comb_data[i][2], '')
				post_recon += cur_post_recon + ' '		
				# print(comb_data[prev_0+1][0])
				# print(entry[0])
				esp_id += ('_%d_%d_' % (char_rem_dict[comb_data[prev_0+1][0]], char_rem_dict[entry[0]]))	
				prev_0 = ind
		# for m in reg_st[att_num_int-1].finditer(post):
		# 	start_ind = m.start() + len(m.group())
		# 	for j in end_indices:
		# 		if j >= start_ind:
		# 			end_ind = j
		# 			break
		# 	# print(post[start_ind:end_ind])
		# 	post_recon += post[start_ind:end_ind] + ' '
		# 	esp_id += ('_%d_%d_' % (start_ind, end_ind))
		post_recon = post_recon.replace(';','').strip()
		esp_id = esp_id[:-1]
		if row['post'] != post_recon:
			print('%s\n' % f_data["metadata"]["post_id"])
			print(post)
			print("--------------------------")
			print(row['post'])
			print("+++++++++++++++++++++++++++++")
			print(post_recon)
			print("******************************")
		# assert(row['post'].encode("utf-8") == post_recon.encode("utf-8"))
		cat_str = ','.join(sorted(list(set([ORG_FOR_LMAP[ORG_LABEL_MAP[cat]] for cat in (row['category_2']).lower().split(',')]))))
		w_fin.writerow({'id': esp_id, 'labels': cat_str})
		# print(esp_id)
		# input()
		# exit()
# for ind, row in enumerate(rows):
# 	row_new = rows_new[ind]
# 	row['category_2'] = ','.join(sorted(row['category_2'].split(',')))
# 	row_new['category_2'] = ','.join(sorted(row_new['category_2'].split(',')))
# 	if row_new != row:
# 		print(row)