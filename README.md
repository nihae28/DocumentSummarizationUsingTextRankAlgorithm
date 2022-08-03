# DocumentSummarizationUsingTextRankAlgorithm
Document Summarization Using TextRankAlgorithm

input dataset:
-------------
The input dataset is hosted on aws s3 bucket with public access.
The required input arguments, articles, and summaries are stored in the following location.
s3://lepuri/final_project/input.txt
s3://lepuri/final_project/articles/
s3://lepuri/final_project/summaries/

Steps:
------
1) Import and execute following code in a databricks notebook.
code url:
https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/2780015303534518/2087200720924936/2314314955150376/latest.html 

2) Install the following using pip
nltk, boto3, networkx, sklearn, bs4, s3fs, rouge

