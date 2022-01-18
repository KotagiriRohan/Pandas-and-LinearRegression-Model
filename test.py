import statsmodels.api as sm

iris = sm.datasets.get_rdataset('mtcars').data
print(iris.index)
