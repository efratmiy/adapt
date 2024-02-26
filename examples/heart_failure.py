import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers.legacy import Adam
from keras.src.layers import Dropout
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from adapt.feature_based import DANN, ADDA, DeepCORAL, CORAL, MCD, MDD, WDGRL, CDAN, SCA, MDA

# load data
df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv")
# split to 4 domains by sex and smoking
# Group by 'sex' and 'smoking'
grouped = df.groupby(['sex', 'smoking'])
# Split into separate DataFrames
dfs = {}  # Dictionary to hold the resulting DataFrames
for (sex, smoking), group_df in grouped:
    dfs[(sex, smoking)] = group_df
Xs = pd.concat([dfs[0, 0], dfs[0, 1], dfs[1, 0]])
# I don't know why they dropped 'creatinine_phosphokinase', but I do the same.
# Xs.drop(['sex', 'smoking', 'creatinine_phosphokinase'], axis=1, inplace=True)
ys = Xs.pop("DEATH_EVENT")
Xt = dfs[1, 1].copy(deep=True)
# Xt.drop(['sex', 'smoking', 'creatinine_phosphokinase'], axis=1, inplace=True)
yt = Xt.pop("DEATH_EVENT")
domains = [0] * len(dfs[0, 0]) + [1] * len(dfs[0, 1]) + [2] * len(dfs[1, 0])

# normalize
std_sc = StandardScaler().fit(Xs)
Xs = std_sc.transform(Xs)
Xt = std_sc.transform(Xt)
# plot input in 2-D
pca = PCA(2).fit(np.concatenate((Xs, Xt)))
Xs_pca = pca.transform(Xs)
Xt_pca = pca.transform(Xt)

x_min, y_min = np.min([Xs_pca.min(0), Xt_pca.min(0)], 0)
x_max, y_max = np.max([Xs_pca.max(0), Xt_pca.max(0)], 0)
x_grid, y_grid = np.meshgrid(np.linspace(x_min - 0.1, x_max + 0.1, 100),
                             np.linspace(y_min - 0.1, y_max + 0.1, 100))
X_grid = pca.inverse_transform(np.stack([x_grid.ravel(), y_grid.ravel()], -1))
domains_df = pd.Series(domains)
ys_array = np.array(ys)
marker_list = ["o", "v", "s"]
color_list = ["r", "b"]

fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))
ax1.set_title("Input space")
for y_value in range(2):
    for domain_value in range(3):
        mask = (domains_df == domain_value) & (ys_array == y_value)
        ax1.scatter(Xs_pca[mask, 0], Xs_pca[mask, 1], label=f"source-domain {domain_value}, class {y_value}",
                    edgecolors='k', c=color_list[y_value], marker=marker_list[domain_value], alpha=0.7)

for y_value in range(2):
    mask = (yt == y_value)
    ax1.scatter(Xt_pca[mask, 0], Xt_pca[mask, 1], label=f"target, class {y_value}",
                marker= "D", edgecolors=color_list[y_value],  facecolors='none', alpha=0.7)
ax1.legend()
ax1.set_yticklabels([])
ax1.set_xticklabels([])
ax1.tick_params(direction='in')
plt.show()


def show_results(model, is_src_only=False, domains=None):
    if is_src_only:
        ys_pred = model.predict_estimator(Xs)
        yt_pred = model.predict_estimator(Xt)
        Xs_enc = Xs
        Xt_enc = Xt
        yp_grid = model.predict_estimator(X_grid).reshape(100, 100)

    else:
        ys_pred = model.predict(Xs, domains)
        yt_pred = model.predict(Xt)
        if domains is None:
            Xs_enc = model.transform(Xs)
        else:
            Xs_enc = model.transform(Xs, domains)
        Xt_enc = model.transform(Xt)
        yp_grid = model.predict(X_grid).reshape(100, 100)


    acc_s = accuracy_score(ys, ys_pred > 0.5)
    acc_t = accuracy_score(yt, yt_pred > 0.5)


    pca_enc = PCA(2).fit(np.concatenate((Xs_enc, Xt_enc)))
    Xs_enc_pca = pca_enc.transform(Xs_enc)
    Xt_enc_pca = pca_enc.transform(Xt_enc)

    cm = plt.cm.RdBu
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.set_title("PCA input space")
    ax1.contourf(x_grid, y_grid, yp_grid, cmap=cm, alpha=0.6)
    for y_value in range(2):
        for domain_value in range(3):
            mask = (domains_df == domain_value) & (ys_array == y_value)
            ax1.scatter(Xs_pca[mask, 0], Xs_pca[mask, 1], label=f"source-domain {domain_value}, class {y_value}",
                        edgecolors='k', c=color_list[y_value], marker=marker_list[domain_value], alpha=0.7)

    for y_value in range(2):
        mask = (yt == y_value)
        ax1.scatter(Xt_pca[mask, 0], Xt_pca[mask, 1], label=f"target, class {y_value}",
                    marker="D", edgecolors=color_list[y_value], facecolors='none', alpha=0.7)
    ax1.legend()
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax1.tick_params(direction='in')


    ax2.set_title("PCA encoded space")
    for y_value in range(2):
        for domain_value in range(3):
            mask = (domains_df == domain_value) & (ys_array == y_value)
            ax2.scatter(Xs_enc_pca[mask, 0], Xs_enc_pca[mask, 1], label=f"source-domain {domain_value}, class {y_value}",
                        edgecolors='k', c=color_list[y_value], marker=marker_list[domain_value], alpha=0.7)

    for y_value in range(2):
        mask = (yt == y_value)
        ax2.scatter(Xt_enc_pca[mask, 0], Xt_enc_pca[mask, 1], label=f"target, class {y_value}",
                    marker="D", edgecolors=color_list[y_value], facecolors='none', alpha=0.7)
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    ax2.tick_params(direction='in')
    if is_src_only:
        fig.suptitle("%s - Source Acc : %.3f - Target Acc : %.3f" % ("Source Only", acc_s, acc_t))
        d = {'name': "no DG", 'Source_Acc': acc_s, 'Target_Acc': acc_t}
    else:
        fig.suptitle("%s - Source Acc : %.3f - Target Acc : %.3f" % (model.__class__.__name__, acc_s, acc_t))
        d = {'name': model.__class__.__name__, 'Source_Acc': acc_s, 'Target_Acc': acc_t}

    return d

def plot_training(model, name):
    pd.DataFrame(model.estimator_.history.history).plot(figsize=(8, 5))
    plt.title(f"{name} Training history", fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Scores")
    plt.legend(ncol=2)


def get_task(activation="sigmoid", units=1):
    # Network
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(10, activation="relu"))
    model.add(Dropout(0.2))
    # model.add(Dense(10, activation="relu"))
    model.add(Dense(units, activation=activation))
    model.compile(optimizer=Adam(0.001), loss='mean_squared_error')
    return model


results = pd.DataFrame()

# no DG
noDG = SCA(copy=True, EIG_RATIO=3, BETA=0.1, DELTA=1000, estimator=get_task(),
          metrics=["acc"], random_state=0)
noDG.fit_estimator(Xs, ys, epochs=200, batch_size=34, verbose=1)
plot_training(noDG, 'noDG')
results = results._append(show_results(noDG, True), ignore_index=True)
#
# # Source Only
# # For source only, we use a DANN instance with lambda set to zero. Thus, the gradient of the discriminator is not back-propagated through the encoder.
# src_only = DANN(task=get_task(),
#                 loss="bce", optimizer=Adam(0.001, beta_1=0.5),
#                 copy=True, lambda_=0., metrics=["acc"], random_state=0)
#
# src_only.fit(Xs, ys, Xt, epochs=200, batch_size=34, verbose=1)
# pd.DataFrame(src_only.history_).plot(figsize=(8, 5))
# plt.title("Training history", fontsize=14)
# plt.xlabel("Epochs")
# plt.ylabel("Scores")
# plt.legend(ncol=2)
# plt.show()
#
# results = results._append(show_results(src_only, True), ignore_index=True)

# MDA
MDA = MDA(copy=True, alph=100, beta=0.1, gamm=10, eig_ratio=3, estimator=get_task(),
          metrics=["acc"], random_state=0)

MDA.fit(Xs, ys, Xt, yt, domains=domains, epochs=200, batch_size=34, verbose=1)
plot_training(MDA, 'MDA')
results = results._append(show_results(MDA, domains=domains), ignore_index=True)

# SCA
SCA = SCA(copy=True, EIG_RATIO=3, BETA=0.1, DELTA=1000, estimator=get_task(),
          metrics=["acc"], random_state=0)

SCA.fit(Xs, ys, Xt, yt, domains=domains, epochs=200, batch_size=34, verbose=1)
plot_training(SCA, 'SCA')
results = results._append(show_results(SCA, domains=domains), ignore_index=True)

# coral
coral = CORAL(estimator=get_task(), lambda_=0.,
              loss="bce", optimizer=Adam(0.001, beta_1=0.5),
              copy=True, metrics=["acc"], random_state=0)
coral.fit(Xs, ys, Xt, epochs=200, batch_size=34, verbose=0, domains=domains);
plot_training(coral, 'coral')
results = results._append(show_results(coral, domains='src'), ignore_index=True)

print(results)

# Get a list of all figure numbers of open figures
fig_nums = plt.get_fignums()
figures = [plt.figure(n) for n in fig_nums]
for i, fig in enumerate(figures):
    fig.savefig(f"C:/Users/efrat/Dropbox/W2B/sca/ADAPT results/heart failure/dropout/figure_{i+1}.png")
plt.show()


# # Source Only
# # For source only, we use a DANN instance with lambda set to zero. Thus, the gradient of the discriminator is not back-propagated through the encoder.
# src_only = DANN(task=get_task(),
#                 loss="bce", optimizer=Adam(0.001, beta_1=0.5),
#                 copy=True, lambda_=0., metrics=["acc"], random_state=0)
#
# src_only.fit(Xs, ys, Xt, epochs=200, batch_size=34, verbose=1, domains=domains)
# pd.DataFrame(src_only.history_).plot(figsize=(8, 5))
# plt.title("Training history", fontsize=14)
# plt.xlabel("Epochs")
# plt.ylabel("Scores")
# plt.legend(ncol=2)
# plt.show()
#
# results = results._append(show_results(src_only, True, domains=domains), ignore_index=True)
#
# # DANN
# dann = DANN(task=get_task(), loss="bce", optimizer=Adam(0.001, beta_1=0.5),
#             copy=True, lambda_=1., metrics=["acc"], random_state=0)
# dann.fit(Xs, ys, Xt, epochs=200, batch_size=34, verbose=0, domains=domains);
# pd.DataFrame(dann.history_).plot(figsize=(8, 5))
# plt.title("Training history", fontsize=14);
# plt.xlabel("Epochs");
# plt.ylabel("Scores")
# plt.legend(ncol=2)
# plt.show()
#
# results = results._append(show_results(dann, domains=domains), ignore_index=True)
# #
# # # ADDA
# adda = ADDA(task=get_task(),
#             loss="bce", optimizer=Adam(0.001, beta_1=0.5),
#             copy=True, metrics=["acc"], random_state=0)
# adda.fit(Xs, ys, Xt, epochs=200, batch_size=34, verbose=0,domains=domains);
# pd.DataFrame(adda.history_).plot(figsize=(8, 5))
# plt.title("Training history", fontsize=14);
# plt.xlabel("Epochs");
# plt.ylabel("Scores")
# plt.legend(ncol=2)
# plt.show()
#
# results = results._append(show_results(adda, domains=domains), ignore_index=True)
# #
# # DeepCORAL
# dcoral = DeepCORAL(encoder=src_only.encoder_,
#                    task=src_only.task_, lambda_=1000.,
#                    loss="bce", optimizer=Adam(0.001, beta_1=0.5),
#                    copy=True, metrics=["acc"], random_state=0,)
# dcoral.fit(Xs, ys, Xt, epochs=200, batch_size=34, verbose=0, domains=domains);
# pd.DataFrame(dcoral.history_).plot(figsize=(8, 5))
# plt.title("Training history", fontsize=14);
# plt.xlabel("Epochs");
# plt.ylabel("Scores")
# plt.legend(ncol=2)
# plt.show()
#
# results = results._append(show_results(dcoral, domains=domains), ignore_index=True)
#
#
# # MCD
# mcd = MCD(task=get_task(),
#           loss="bce", optimizer=Adam(0.001, beta_1=0.5),
#           copy=True, metrics=["acc"], random_state=0)
# mcd.fit(Xs, ys, Xt, epochs=200, batch_size=34, verbose=0, domains=domains);
# pd.DataFrame(mcd.history_).plot(figsize=(8, 5))
# plt.title("Training history", fontsize=14);
# plt.xlabel("Epochs");
# plt.ylabel("Scores")
# plt.legend(ncol=2)
# plt.show()
#
# results = results._append(show_results(mcd, domains=domains), ignore_index=True)
#
# # MDD
# mdd = MDD(task=get_task(),
#           loss="bce", optimizer=Adam(0.001, beta_1=0.5),
#           copy=True, metrics=["acc"], random_state=0)
# mdd.fit(Xs, ys, Xt, epochs=200, batch_size=34, verbose=0, domains=domains);
# pd.DataFrame(mdd.history_).plot(figsize=(8, 5))
# plt.title("Training history", fontsize=14);
# plt.xlabel("Epochs");
# plt.ylabel("Scores")
# plt.legend(ncol=2)
# plt.show()
#
# results = results._append(show_results(mdd, domains=domains), ignore_index=True)
#
# # WDGRL
# wdgrl = WDGRL(task=get_task(None), gamma=0.,
#               loss="bce", optimizer=Adam(0.001, beta_1=0.5),
#               copy=True, metrics=["acc"], random_state=0)
# wdgrl.fit(Xs, ys, Xt, epochs=200, batch_size=34, verbose=0, domains=domains);
# pd.DataFrame(wdgrl.history_).plot(figsize=(8, 5))
# plt.title("Training history", fontsize=14);
# plt.xlabel("Epochs");
# plt.ylabel("Scores")
# plt.legend(ncol=2)
# plt.show()
#
# results = results._append(show_results(wdgrl, domains=domains), ignore_index=True)
#
# # CDAN
# cdan = CDAN(task=get_task(), entropy=True,
#             loss="bce", optimizer=Adam(0.001, beta_1=0.5),
#             copy=True, random_state=0)
# cdan.fit(Xs, ys, Xt, epochs=200, batch_size=34, verbose=0, domains=domains);
# pd.DataFrame(cdan.history_).plot(figsize=(8, 5))
# plt.title("Training history", fontsize=14);
# plt.xlabel("Epochs");
# plt.ylabel("Scores")
# plt.legend(ncol=2)
# plt.show()
#
# results = results._append(show_results(cdan, domains=domains), ignore_index=True)
