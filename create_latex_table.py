import pandas as pd
import os
from os.path import join

root = './logs'

build_id = 'BUILD'
pass_id = 'PASS'
derpp_id = 'DER++'

cifar10_5T_id = 'CIFAR10-5T'
cifar100_10T_id = 'CIFAR100-10T'
cifar100_20T_id = 'CIFAR100-20T'

vocab = {'sm_scores': 'SM',
         'smmd_scores': 'SMMD',
         'en_scores': 'EN',
         'enmd_scores': 'ENMD'}


model_dataset_exp_dict = {
    derpp_id  : {
        cifar10_5T_id: 'derpp_cifar10-5T',  # canon one
        cifar100_10T_id: 'derpp_cifar10-5T',
        cifar100_20T_id: 'derpp_cifar10-5T',
    },
    build_id  : {
        cifar10_5T_id: 'derpp_cifar10-5T',  # canon one
        cifar100_10T_id: 'derpp_cifar10-5T',
        cifar100_20T_id: 'derpp_cifar10-5T',
    },
}

metrics = ['ACA', 'AIA', 'AF', 'AUC', 'AUPR']

model_list = [build_id, derpp_id]
dataset_list = [cifar10_5T_id, cifar100_10T_id, cifar100_20T_id]

for model in model_list:
    dfs = []
    for ds in dataset_list:
        exp = model_dataset_exp_dict[model][ds]
        csv_path = join(root, exp, 'full_results.csv')
        df = pd.read_csv(csv_path).drop(['Unnamed: 0'],axis=1)
        df['scorer'] = df['scorer'].str.split('_').str[0].str.upper()
        df['detector'] = df['detector'].str.capitalize()
        df = df.round(2)


        avg = df[metrics].mean()
        avg["detector"] = "Avg."
        avg["scorer"] = ""
        avg = avg[df.columns]   # use same order
        avg = avg.astype(str)

        for m in metrics:
            series = df[m]
            series_str = df[m].astype(str)
            idx = series.argmin() if m == 'AF' else series.argmax()
            series_str[idx] = "\\textbf{" + series_str[idx] + "}"
            df[m] = series_str
        
        

        

        scorers = df["scorer"].copy().to_numpy()

        for i in range(len(scorers)):
            if i == 0 or scorers[i] != scorers[i-1]:
                latex_str = f"\\multirow{{4}}{{*}}{{\\rotatebox{{90}}{{\\textbf{{{scorers[i]}}}}}}}"
                df.loc[i, 'scorer'] = latex_str
            else:
                df.loc[i, 'scorer'] = ""

        df["detector"] = df["detector"].apply(lambda x: f"\\textbf{{{x}}}")
        
        df = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)

        dfs.append(df)

        print("")

    metric_parts = [df[metrics].reset_index(drop=True) for df in dfs]
    fixed_part = dfs[0][['scorer', 'detector']].reset_index(drop=True)
    df_combined = pd.concat([fixed_part] + metric_parts, axis=1)

    df_combined.to_latex("table.tex", index=False, escape=False)
    print("")