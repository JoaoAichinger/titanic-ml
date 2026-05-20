import src.data_loader as dl
import src.processor as pr
import src.trainer as tr
import pandas as pd

df = dl.load_data('data/titanic.csv')
df = pr.process(df)
score = tr.train(df)

print(score)
