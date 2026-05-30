import src.data_loader as dl
import src.preprocessor as pr
import src.trainer as tr
import pandas as pd

df = dl.load_data('data/titanic.csv')
df = pr.preprocess(df)
score = tr.forest_train(df)

print(score)
