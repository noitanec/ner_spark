import sparknlp
from ner import NER

spark = sparknlp.start()
model = NER()

with open('../data/news.txt', 'r') as f:
    data = f.read().replace('\n', '').replace(',', ' ').split('.')

df = spark.createDataFrame([[x] for x in data]).toDF('text')
results = model.predict(df).collect()
res_txt = ""
for result in results:
    txt = result['text']
    try:
        ent = result['entities'][0]['result']
        ner = result['entities'][0]['metadata']['entity']
    except IndexError as e:
        ent, ner = '', ''
        print(txt)
    
    newline = '\n' if res_txt  else ''
    res_txt += f"{newline}{txt}, {ent}, {ner}"

with open('../data/output.csv', 'w') as f:
    f.write(res_txt)