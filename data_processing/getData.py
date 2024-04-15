from leetcode import getLeetcode
from codeforces import getCodeforces
from hackerearth import getHackerEarth
from interviewbit import getInterviewBit
import json


doc1, link1 = getLeetcode()
print("Leetcode completed")
doc2, link2 = getCodeforces()
print("Codeforces completed")
doc3, link3 = getInterviewBit()
print("Interviewbit completed")
doc4, link4 = getHackerEarth()
print("Hackerearth completed")


documents = doc1 + doc2 + doc3 + doc4
links = link1 + link2 + link3 + link4

with open('documents.txt', 'w') as f:
  json.dump(documents, f, ensure_ascii=False)

with open('links.txt', 'w') as f:
  json.dump(links, f, ensure_ascii=False)

