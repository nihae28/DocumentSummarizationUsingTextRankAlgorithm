pip install  nltk  boto3  networkx  sklearn  bs4  pandas  s3fs


pip install boto3 --upgrade



pip install rouge --upgrade



pip install bs4 requests


pip install networkx --upgrade


pip install --upgrade scipy networkx



pip install boto3 --upgrade



pip install rouge-score  rouge-metric



import  rouge
import   nltk
import  s3fs
import  networkx  as  ntwrx

import   pandas
import  requests

import  matplotlib.pyplot  as  pyplt
import  boto3

nltk.download('stopwords')
from  sklearn.metrics.pairwise  import  cosine_similarity
from   nltk.tokenize.punkt  import   *
from  pyspark.sql.types   import   *



!wget   http://nlp.stanford.edu/data/glove.6B.zip

!unzip   glove*.zip



input_args = spark.read.text("s3://bdmayuva/final_project/input.txt")
input_list = input_args.rdd.flatMap(list)
input_list = input_list.collect()
articles_path = input_list[0]
summaries_path = input_list[1]

print(articles_path)
print(summaries_path)




def getEmbeddingsForWords():
    fname = 'glove.6B.100d.txt'
    encoding_type = 'utf-8'
    type_d = 'float32'
    
    file_ptr = open (fname, encoding = encoding_type)

    for ln in file_ptr:
        val = ln.split()
        wds = val[0]
        cfs = nmpy.asarray (val [1:], dtype = type_d)
        embdgs_for_words[wds] = cfs
        
    file_ptr.close()



import  networkx  as  ntwrx

def cosineSimilarity(matrix):
  sum_value = nmpy.sum(0, keepdims=True)
  norm_matrix = math.power((matrix.T * matrix.T).sum_value, 0.5)
  return (matrix @ matrix.T) / norm_matrix .T / norm_matrix 

def tokensGet(input_sentence):
    split_line = ' '.join(input_sentence.strip().split('\n'))
    sent_tknizer = PunktSentenceTokenizer()
    result_token_sentence = sent_tknizer.tokenize(split_line)
    return result_token_sentence

def rankSentences(sentences, similarityMatrix):
    sentences_graph = ntwrx.from_numpy_array(similarityMatrix)
    sentences_scores = ntwrx.pagerank(sentences_graph)
    sentences_ranked = sorted(((sentences_scores[idx],idx) for idx, sntns in enumerate(sentences)), reverse=True)
    
    return sentences_ranked



def eliminate_stopwords(raw_sentence):
  stopwords_eliminated_sentence = [idx for idx in raw_sentence if idx not in e_stp_wrds]
  return stopwords_eliminated_sentence


#similarity matrix
def calculateSimilarityMatrix(vectors, sentences, dim):
    sentence_length = len(sentences)
    similarity_matrix = nmpy.zeros([sentence_length,sentence_length])
        
    for x in range(sentence_length):
        for y in range(sentence_length):
            if x != y:
                similarity_matrix[x][y] = cosine_similarity(vectors[x].reshape(1,dim), vectors[y].reshape(1,dim))[0,0]
                
    return similarity_matrix

def extract_summary(ranked_sentences, sentence_number):

  result_lst = []
  sentences_summary = min(sentence_number, len(ranked_sentences))
  for idx in range(sentences_summary):
    result_lst.append ( ranked_sentences[idx][1])
    
  return result_lst



# DBTITLE 1,MAIN CODE STARTS HERE
# MAIN FUNCTION
from nltk.corpus import stopwords
import numpy as nmpy

embdgs_for_words = {}
#Get Embeddings for words
getEmbeddingsForWords()

e_stp_wrds = set (stopwords.words ('english'))
input_file = sc.wholeTextFiles(articles_path)

file_sentences = input_file.map(lambda x: (x[0], tokensGet(x[1]))).collect()

processed_sentences2 = []

i=0
while i < len(file_sentences):
  id = file_sentences[i][0]
  text = file_sentences[i][1]
  processed_sentences2.append((id,eliminate_stopwords(text)))
  i+=1

# print(processed_sentences2)



from rouge_score import rouge_scorer
input_df1 = sc.wholeTextFiles(summaries_path).toDF().toDF("id", "doc")
summaries_text = input_df1.toPandas()

ref=[]

all_precision_l=[]
all_recall_l=[]
all_fmeasure_l=[]

all_precision_1=[]
all_recall_1=[]
all_fmeasure_1=[]

all_precision_2=[]
all_recall_2=[]
all_fmeasure_2=[]

for idx, summ in summaries_text.iterrows():
  ref1 = summ['doc']
  ref.append(ref1)

num_sentences = [num for num in range(1, 12)]

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

itr = 0
for line in processed_sentences2:
  summary = line[0]
  text = line[1]
  sentence_list = []
  
  current_precision_l=[]
  current_recall_l=[]
  current_fmeasure_l=[]
  
  current_precision_1=[]
  current_recall_1=[]
  current_fmeasure_1=[]
  
  current_precision_2=[]
  current_recall_2=[]
  current_fmeasure_2=[]

  for i in text:
    v = sum([embdgs_for_words.get(wrd, nmpy.zeros((100,))) for wrd in i.split()])/(len(i.split())+0.001)
    sentence_list.append(nmpy.array(v))

  sentence_list = nmpy.array(sentence_list)
  sim_mat = calculateSimilarityMatrix(sentence_list,text,100)

  for k1 in range(1,len(num_sentences)+1,1):
    ranked_sentences = rankSentences(text, sim_mat)
    result_lst = extract_summary(ranked_sentences, k1)
    a = [text[i] for i in result_lst]
    summary = ''.join(a)
    
#     print(summary)
  
    score = scorer.score(summary, ref[itr])
    
    precision_l, recall_l, fmeasure_l = score['rougeL']
    current_precision_l.append(precision_l)
    current_recall_l.append(recall_l)
    current_fmeasure_l.append(fmeasure_l)
    
    precision_1, recall_1, fmeasure_1 = score['rouge1']
    current_precision_1.append(precision_1)
    current_recall_1.append(recall_1)
    current_fmeasure_1.append(fmeasure_1)
    
    precision_2, recall_2, fmeasure_2 = score['rouge2']
    current_precision_2.append(precision_2)
    current_recall_2.append(recall_2)
    current_fmeasure_2.append(fmeasure_2)
  
  all_precision_l.append(current_precision_l)
  all_recall_l.append(current_recall_l)
  all_fmeasure_l.append(current_fmeasure_l)
  
  all_precision_1.append(current_precision_1)
  all_recall_1.append(current_recall_1)
  all_fmeasure_1.append(current_fmeasure_1)
  
  all_precision_2.append(current_precision_2)
  all_recall_2.append(current_recall_2)
  all_fmeasure_2.append(current_fmeasure_2)

  itr = itr + 1

#print(all_precision)



# Plotting Graphs for the various Metrics
def plot_graphs (metric_type, label_type, num, legend):
    file_name = label_type+".png"
    itr = 1
    color = 'g--'
    if label_type == 'precision':
      color = 'r--'
    elif label_type == 'recall':
      color = 'b--'
      
    pyplt.figure(figsize=(8,8))
    for metric in metric_type:
      pyplt.plot(num, metric, color, label = label_type)
      itr = itr + 1

    pyplt.xlabel("Number of Sentences")
    pyplt.ylabel(label_type)
    pyplt.legend([legend])
    pyplt.show()
    pyplt.savefig(file_name)



plot_graphs (all_precision_l, "precision", num_sentences, "ROUGE-L")
plot_graphs (all_recall_l, "recall", num_sentences, "ROUGE-L")
plot_graphs (all_fmeasure_l, "fmeasure", num_sentences, "ROUGE-L")



plot_graphs (all_precision_1, "precision", num_sentences, 'ROUGE-1')
plot_graphs (all_recall_1, "recall", num_sentences, 'ROUGE-1')
plot_graphs (all_fmeasure_1, "fmeasure", num_sentences, 'ROUGE-1')



plot_graphs (all_precision_2, "precision", num_sentences, 'ROUGE-2')
plot_graphs (all_recall_2, "recall", num_sentences, 'ROUGE-2')
plot_graphs (all_fmeasure_2, "fmeasure", num_sentences, 'ROUGE-2')




