import os
import re

# st = '<h2 class="entry-title"><a href="https://everydaysexism.com/everyday-sexism/176976" rel="bookmark" title="AG">AG</a></h2>does not\nmatter<h2 class="entry-title"><a href="https://everydaysexism.com/everyday-sexism/00000176976" rel="bookmark" title="Shori">Sh</a></h2>\ndamn'
# re1 = re.compile(r'(<\s*h2\s*class\s*=\s*"entry-title"\s*><a.+title\s*=\s*")(.*)("\s*>)(.*)(</a></h2>)')
# print(re1.sub(r'\1REDACTED\3REDACTED\5', st))
# exit()
re1 = re.compile(r'(<\s*h2\s*class\s*=\s*"entry-title"\s*><a.+title\s*=\s*")(.*)("\s*>)(.*)(</a></h2>)')

f=open('file9.txt','r')
content = f.read()
f.close()

f = open('file9_copy.txt','w')
f.write(re1.sub(r'\1REDACTED\3REDACTED\5', content))
f.close()

# for line in text:
# 	if 'title' in line:
# 		res = line.split("=")
# 		if len(res) == 6 :
# 			res[4]= "'REDACTED'>REDACTED</a></h2> <section class"
# 			line = res[0] + "=" +res[1] + "=" + res[2] + "=" + res[3] + "=" + res[4] + "=" + res[5]
# 			f.write(line)
# 		else:
# 			f.write(line)
# 	else:
# 		f.write(line)
