import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
import pdb

file = open('../model_outputs/xgboost/xgb_model.pkl', 'rb')
model = pickle.load(file)
file.close()
# pdb.set_trace()
# sorted_idx = xgb.feature_importances_.argsort()
xgb.plot_tree(model, num_trees=4)
xgb.plot_importance(model, max_num_features=20)
# plt.barh(model.feature_names[sorted_idx], xgb.feature_importances_[sorted_idx])
plt.show()