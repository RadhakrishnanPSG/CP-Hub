import requests
import pandas as pd
from stopwords import remove_stopwords

def getLeetcode():
	lc = requests.get("https://leetcode.com/api/problems/all/")
	lc_json = lc.json()

	q_id = []
	q_name = []
	question__title_slug = []
	difficulty_level = []
	for i in lc_json['stat_status_pairs']:
		q_id.append(i['stat']['question_id'])
		q_name.append(i['stat']['question__title'])
		question__title_slug.append(i['stat']['question__title_slug'])
		difficulty_level.append(i['difficulty']['level'])

	lc_data = pd.DataFrame(list(zip(q_id, q_name, question__title_slug, difficulty_level)), columns =['q_id', 'q_name', 'question__title_slug', 'level'])

	words1 = list(lc_data['question__title_slug'])
	words2 = list(lc_data['q_name'])
	tokens = []
	temp = []
	links = []


	for word1, word2 in zip(words1, words2):
		links.append(['https://leetcode.com/problems/'+word1+'/'])
		tokens.append(word2.split() + ["leetcode"])

	tokens = remove_stopwords(tokens)
	return tokens, links




