# In[] Libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro, normaltest, anderson, mannwhitneyu

# In[] Dataset
_cols = ["ID", "Age", "Gender", "Weight", "Height", "BMI", "PA Level"]

_group_size, _age_m, _age_std, _weight_m, _weight_std, _height_m, _height_std = 50, 70, 9, 80, 20, 175, 50
_array_dataset = np.zeros(len(_cols))

for _index in range(_group_size):
    _id = f"P{_index}"
    _age = int(np.random.randn(1)[0] * _age_std + _age_m)
    _gender = "M"
    _weight = int(np.random.randn(1)[0] * _weight_std + _weight_m)
    _height = int(np.random.randn(1)[0] * _height_std + _height_m)
    _BMI = np.round(_weight/(_height/100)**2, 2)
    _PA_level = np.random.randn(1)[0] + _BMI/2
    _array_dataset = np.vstack((_array_dataset, [_id, _age, _gender, _weight, _height, _BMI, _PA_level]))

df = pd.DataFrame(
    data=_array_dataset[1:, :],
    columns=_cols,
)

df["Age"] = pd.to_numeric(df["Age"])
df["Weight"] = pd.to_numeric(df["Weight"])
df["Height"] = pd.to_numeric(df["Height"])
df["BMI"] = pd.to_numeric(df["BMI"])
df["PA Level"] = pd.to_numeric(df["PA Level"])
print(df.describe())

# In[] add some outliers
_id, _age, _weight, _height = _index+1, 100, 50, 120
_array_dataset = np.vstack((_array_dataset, [f"P{_id}", _age, "M", _weight, _height, np.round(_weight/(_height/100)**2, 2), np.random.randn(1)[0] + np.round(_weight/(_height/100)**2, 2)/2]))

_id, _age, _weight, _height = _index+2, 95, 52, 90
_array_dataset = np.vstack((_array_dataset, [f"P{_id}", _age, "M", _weight, _height, np.round(_weight/(_height/100)**2, 2), np.random.randn(1)[0] + np.round(_weight/(_height/100)**2, 2)/2]))

df2 = pd.DataFrame(
    data=_array_dataset[1:, :],
    columns=_cols,
)

df2["Age"] = pd.to_numeric(df2["Age"])
df2["Weight"] = pd.to_numeric(df2["Weight"])
df2["Height"] = pd.to_numeric(df2["Height"])
df2["BMI"] = pd.to_numeric(df2["BMI"])
df2["PA Level"] = pd.to_numeric(df2["PA Level"])
print(df2.describe())

# In[] add a cluster of outliers
_group_size, _age_m, _age_std, _weight_m, _weight_std, _height_m, _height_std = 50, 95, 15, 90, 5, 160, 5
_array_dataset3 = np.zeros(len(_cols))

for _index in range(_group_size):
    _id = f"P{_index}"
    _age = int(np.random.randn(1)[0] * _age_std + _age_m)
    _gender = "M"
    _weight = int(np.random.randn(1)[0] * _weight_std + _weight_m)
    _height = int(np.random.randn(1)[0] * _height_std + _height_m)
    _BMI = np.round(_weight/(_height/100)**2, 2)
    _PA_level = np.random.randn(1)[0] + _BMI/2
    _array_dataset3 = np.vstack((_array_dataset3, [_id, _age, _gender, _weight, _height, _BMI, _PA_level]))

_array_dataset3 = np.vstack((_array_dataset, _array_dataset3))

df3 = pd.DataFrame(
    data=_array_dataset3[1:, :],
    columns=_cols,
)

df3["Age"] = pd.to_numeric(df3["Age"])
df3["Weight"] = pd.to_numeric(df3["Weight"])
df3["Height"] = pd.to_numeric(df3["Height"])
df3["BMI"] = pd.to_numeric(df3["BMI"])
df3["PA Level"] = pd.to_numeric(df3["PA Level"])
print(df3.describe())

# In[] Plot the dataset
_cols_to_analyse = ["Age", "Weight", "Height", "BMI", "PA Level"]
plt.violinplot(df[_cols_to_analyse], showmeans=True, showmedians=True)
plt.violinplot(df2[_cols_to_analyse], showmeans=True, showmedians=True)
plt.violinplot(df3[_cols_to_analyse], showmeans=True, showmedians=True)
plt.xticks(ticks=range(1, len(_cols_to_analyse)+1), labels=_cols_to_analyse)
plt.show()

# In[] Normality Test
_col = "Age"
print("*" * 10, "DF1", "*" * 10)
_data = df[_col]
stat, p = shapiro(_data) # Shapiro-Wilk Test+
9
print("Statistics=%.3f, p=%.3f" % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print("Sample looks Gaussian (fail to reject H0)")
else:
	print("Sample does not look Gaussian (reject H0)")
    
stat, p = normaltest(_data) # D’Agostino’s K^2 Test
print("Statistics=%.3f, p=%.3f" % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print("Sample looks Gaussian (fail to reject H0)")
else:
	print("Sample does not look Gaussian (reject H0)")

result = anderson(_data) # Anderson-Darling Test
print("Statistic=%.3f" % result.statistic)
p = 0
for i in range(len(result.critical_values)):
	sl, cv = result.significance_level[i], result.critical_values[i]
	if result.statistic < result.critical_values[i]:
		print("sigL: %.3f: cVal: %.3f, data looks normal (fail to reject H0)" % (sl, cv))
	else:
		print("sigL: %.3f: cVal: %.3f, data does not look normal (reject H0)" % (sl, cv))

print("*" * 10, "DF2", "*" * 10)
_data = df2[_col]
stat, p = shapiro(_data) # Shapiro-Wilk Test
print("Statistics=%.3f, p=%.3f" % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print("Sample looks Gaussian (fail to reject H0)")
else:
	print("Sample does not look Gaussian (reject H0)")
    
stat, p = normaltest(_data) # D’Agostino’s K^2 Test
print("Statistics=%.3f, p=%.3f" % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print("Sample looks Gaussian (fail to reject H0)")
else:
	print("Sample does not look Gaussian (reject H0)")

result = anderson(_data) # Anderson-Darling Test
print("Statistic=%.3f" % result.statistic)
p = 0
for i in range(len(result.critical_values)):
	sl, cv = result.significance_level[i], result.critical_values[i]
	if result.statistic < result.critical_values[i]:
		print("sigL: %.3f: cVal: %.3f, data looks normal (fail to reject H0)" % (sl, cv))
	else:
		print("sigL: %.3f: cVal: %.3f, data does not look normal (reject H0)" % (sl, cv))

print("*" * 10, "DF3", "*" * 10)
_data = df3[_col]
stat, p = shapiro(_data) # Shapiro-Wilk Test
print("Statistics=%.3f, p=%.3f" % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print("Sample looks Gaussian (fail to reject H0)")
else:
	print("Sample does not look Gaussian (reject H0)")
    
stat, p = normaltest(_data) # D’Agostino’s K^2 Test
print("Statistics=%.3f, p=%.3f" % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print("Sample looks Gaussian (fail to reject H0)")
else:
	print("Sample does not look Gaussian (reject H0)")

result = anderson(_data) # Anderson-Darling Test
print("Statistic=%.3f" % result.statistic)
p = 0
for i in range(len(result.critical_values)):
	sl, cv = result.significance_level[i], result.critical_values[i]
	if result.statistic < result.critical_values[i]:
		print("sigL: %.3f: cVal: %.3f, data looks normal (fail to reject H0)" % (sl, cv))
	else:
		print("sigL: %.3f: cVal: %.3f, data does not look normal (reject H0)" % (sl, cv))
        
# In[] Hypothesis Test
stat, p = mannwhitneyu(data1, data2)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Same distribution (fail to reject H0)')
else:
	print('Different distribution (reject H0)')