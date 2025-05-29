import os
from experiments import *

base_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))



exp = Experiment(base_project_dir, 0, 'Transformer', exp_name = '10fold-32channel')
exp.prepare()
#exp.train_10fold()
exp.reproduce_results_on_10fold(evaluation_set='test', plot_attention_weights=True, plot_self_attention_weights=False)
#exp.evalute_on_PRED_CT(model_number=5)
#exp.evalute_on_UI(model_number=5)

#exp.train_10fold_SIT()
#exp.reproduce_results_on_10fold_SIT(evaluation_set='val', plot_attention_weights=False)




