import xgboost as xgb
import pandas as pd
import numpy as np
import itertools
import random
import pickle 
import lightgbm as lgb
import matplotlib.pyplot as pl
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
pl.tick_params(labelsize=6)
using_cols = ['ID_code', 'target', 'var_0', 'var_1', 'var_2', 'var_5', 'var_6', 'var_9', 'var_13', 'var_18', 'var_19', 'var_21',
			  'var_22', 'var_26', 'var_35', 'var_36', 'var_40', 'var_44', 'var_48', 'var_49', 'var_53', 'var_55',
			  'var_56', 'var_67', 'var_76', 'var_81', 'var_80', 'var_86', 'var_89', 'var_94', 'var_95', 'var_99',
			  'var_109', 'var_110', 'var_115', 'var_119', 'var_123', 'var_127', 'var_135', 'var_139', 'var_146',
			  'var_151', 'var_154', 'var_155', 'var_157', 'var_163', 'var_165', 'var_170', 'var_174', 'var_179',
			  'var_180', 'var_184', 'var_188', 'var_190', 'var_191', 'var_196', 'var_198']
most_important = ['var_13', 'var_34', 'var_108', 'var_94', 'var_53', 'var_80', 'var_9', 'var_184']
def calc_stats(df, skip=2):
	sums = df.ix[:, skip:].sum(axis=1)
	means = sums / sums.shape[-1]
	mins = df.ix[:, skip:].min(axis=1)
	maxs = df.ix[:, skip:].max(axis=1)
	#stds = (((df.ix[:, 2:] - means)**2).sum(axis=1) / sums.shape[-1])**0.5
	sums.columns = ['sum']
	means.columns = ['mean']
	mins.columns = ['min']
	maxs.columns = ['max']
	#stds.columns = ['std']
	#ret = df.join([sums, means, mins, maxs])

	return [sums, means, mins, maxs]

def calc_poly(df, skip=0):
	double = df.ix[:, most_important] ** 2
	triple = df.ix[:, most_important] ** 3
	double.columns = [name + '^2' for name in double.columns]
	triple.columns = [name + '^3' for name in triple.columns]

	return [double, triple]

def calc_round(df, skip=2):
	round1 = df.ix[:, most_important].round(1)
	round2 = df.ix[:, most_important].round(2)
	round1.columns = [name + '_1' for name in round1.columns]
	round2.columns = [name + '_2' for name in round2.columns]

	return [round1, round2]


def normalize(df, skip=2):
	for key in df.columns[skip:]:
		df.ix[:, key] = StandardScaler().fit_transform(df.ix[:, key].values.reshape(-1, 1))

	return df

train_df = pd.read_csv('train.csv')
using_cols.remove('target')
test_df = pd.read_csv('test.csv')
'''
print('calculating round ...')
rounds_train = calc_poly(train_df, skip=2)
rounds_test = calc_poly(test_df, skip=1)
print('calculating poly')
poly_train = calc_stats(train_df, skip=2)
poly_test = calc_stats(test_df, skip=1)
'''

subs = []
for name in most_important:
	rem = [n for n in most_important if n != name]
	sub_feature = train_df.ix[:, rem] - train_df.ix[:, name]
	subs.append(sub_feature)


print('joining')
#train_df = train_df.join([*rounds_train, *poly_train])
#test_df = test_df.join([*rounds_test, *poly_test])



'''
pca = PCA(n_components=100)
print('fitting ...')
pca.fit(train_df.ix[:, 2:].values)
print('done')
train_pca = pd.DataFrame(pca.transform(train_df.ix[:, 2:].values), columns=['pca_{}'.format(i) for i in range(100)])
test_pca = pd.DataFrame(pca.transform(test_df.ix[:, 1:].values), columns=['pca_{}'.format(i) for i in range(100)])
train_df = train_df.join(train_pca)
test_df = test_df.join(test_pca)
'''
#train_df = add_stats(train_df)
#test_df = add_stats(test_df)





all_index = list(range(train_df.shape[0]))
pos_index = [i for i, d in enumerate(train_df.values) if d[1] == 1]
neg_index = [i for i, d in enumerate(train_df.values) if d[1] == 0]
train_index = pos_index[:int(len(pos_index)*0.8)] + neg_index[:-int(len(pos_index)*0.2)]
#train_index = pos_index + neg_index
pos_weight = (len(train_index) - int(len(pos_index))) / int(len(pos_index))
val_index = pos_index[int(len(pos_index)):] + neg_index[-int(len(pos_index)):]

random.shuffle(train_index)

train_data_extend = pd.DataFrame([train_df.ix[i, :] for i in train_index], index=list(range(len(train_index))))
val_data = pd.DataFrame([train_df.ix[i, :] for i in val_index], index=list(range(len(val_index))))
print(train_data_extend.shape)

#train_data = xgb.DMatrix(train_data_extend.ix[:, 2:], label=train_data_extend.ix[:, 'target'])
#val_data = xgb.DMatrix(val_data.ix[:, 2:], label=val_data.ix[:, 'target'])
#test_data = xgb.DMatrix(test_df.ix[:, 1:])

train_data = lgb.Dataset(train_data_extend.ix[:, 2:], label=train_data_extend.ix[:, 'target'])
val_data = lgb.Dataset(val_data.ix[:, 2:], label=val_data.ix[:, 'target'])
test_data = test_df.ix[:, 1:].values

'''
params = {'eta': [0.1, 0.3, 0.5],
		  'gamma': [0, 0.5, 1],
		  'max_depth': [3, 6],
		  'lambda': [0, 0.5, 1],
		  'alpha': [0, .5, 1],
		  'subsample': [0.5, 0.7, 1]}



params_keys = list(params.keys())
search_space = [params[key] for key in params_keys]
results = {}

for items in itertools.product(*search_space):
	print('start case: {}'.format(items))
	param = {key: item for key, item in zip(params, items)}
	param.update({'scale_pos_weight': len(neg_index) / len(pos_index),
				  'eval_metric': 'auc',
				  'verbose_eval': True,
				  'nthread': 8,
				  'predictor': 'gpu_predictor'})
	result = xgb.cv(param, train_data, num_boost_round=100, nfold=5, stratified=True,
					 metrics='auc')
	results[tuple(items)] = max(result['test-auc-mean'].values.flatten())

pickle.dump(results, open('results_dict.pkl', 'wb'))


params = {'eta': 0.5, 'gamma': 1, 'max_depth': 3, 'lambda': 1, 'alpha': 0.5, 'scale_pos_weight': pos_weight,
          'objective': 'binary:logistic', 'subsample': 1, 'eval_metric': 'auc'}
'''
'''
param = {'num_leaves': 31, 'num_trees': 500, 'learning_rate': 0.1,
		 'objective': 'binary', 'metric': 'auc', 'is_unbalance': True}
'''
param = {
    'bagging_freq': 5,
    'bagging_fraction': 0.4,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.05,
    'learning_rate': 0.01,
    'max_depth': -1,  
    'metric':'auc',
    'min_data_in_leaf': 80,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 13,
    'num_threads': 8,
    'tree_learner': 'serial',
    'objective': 'binary', 
    'verbosity': 1
}
#booster = xgb.train(params, train_data, evals=[(val_data, 'eval')], num_boost_round=200, verbose_eval=True, early_stopping_rounds=20)
#preds = booster.predict(test_data)

'''
bst = lgb.train(param, train_data, 8500, valid_sets=[val_data], verbose_eval=1000)

feature_importance = bst.feature_importance()
feature_importance = sorted([(v, n) for v, n in zip(feature_importance, train_df.columns[2:])], key=lambda x: -x[0])[:100]

sns.set()
sns.barplot(y=[e[1] for e in feature_importance], x=[e[0] for e in feature_importance])
pl.show()
'''

res = lgb.cv(param, train_data, 10000, nfold=5, verbose_eval=500)


'''
preds = bst.predict(test_data)

#preds = [int(x >= .5) for x in preds]

with open('submission.csv', 'w') as f:
	f.write('ID_code,target\n')
	for id_, t in zip(test_df.ix[:, 'ID_code'].values, preds):
		f.write('{},{}\n'.format(id_, t))
'''

