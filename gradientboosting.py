## helpful resource for sklearn GBM
## https://www.kaggle.com/beagle01/prediction-with-gradient-boosting-classifier

## for lightgm 
## https://lightgbm.readthedocs.io/en/latest/

import lightgbm as lgb
mdl = lgb.LGBMClassifier(max_depth=-1, min_child_samples=400, 
              random_state=314, silent=True, metric='None', 
              n_jobs=4, n_estimators=500, learning_rate=0.1,
              **{'colsample_bytree': 0.75, 'min_child_weight': 1, 
               'num_leaves': 60, 'subsample': 0.75}
             )

train_file = 'data/train_data.csv'
track_file = 'data/track_feats.csv'
train = pd.read_csv('data/train_data.csv')
track_feats = pd.read_csv('data/track_feats.csv')

def learning_rate_decay_power_0995(current_iter):
    base_learning_rate = 0.15
    lr = base_learning_rate  * np.power(.998, current_iter)
    return lr if lr > 1e-2 else 1e-2


n_fit = None

X = pd.concat([X_trn, X_trk[0]], axis=1)
fit_params = {'eval_names': ['train', 'early_stop'],
              'eval_set': [(X.loc[id_trn,:], y_trn[0].loc[id_trn]), 
                           (X.loc[id_stp,:], y_trn[0].loc[id_stp])],
              'eval_metric': 'binary_error',
              'verbose':50, 'early_stopping_rounds':60,
              'callbacks':[lgb.reset_parameter(learning_rate=learning_rate_decay_power_0995)]}

mdl.fit(X.loc[id_trn,:].iloc[:n_fit], y_trn[0].loc[id_trn].iloc[:n_fit], **fit_params)

prob_pred = mdl.predict_proba(X.loc[id_stp,:])[:,1]
evaluate_model(prob_pred>0.55, y_lists_stp)