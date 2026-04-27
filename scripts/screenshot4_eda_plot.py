from GlassBox.numpandas import read_csv
from GlassBox.eda import plot_manager

df = read_csv("test_data/test_model.csv")
plot_manager.histplot(df, column="feature1", bins=20)