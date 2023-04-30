import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plot_utils
import cpp_utils
import seaborn as sns
import time

import matplotlib as mpl
cmap = plot_utils.cmap 
from matplotlib.lines import Line2D



# def plot_alpha_search(filename="gradientSearch", D=3, beta=1.0, alpha_range=(0.5, 0.5, 1), save=False, interacting=False):
#     alphas = np.linspace(*alpha_range)
#     Ns = [10, 50, 100]
#     stepLengths = [0.55]#, 0.1, 0.5, 1.0]
#     epsilon = 0.01
#     lr= 0.002
#     logMet = 20 # 2 ^ 18 = 262144
#     logEq = 20 # 2 ^ 16 = 65536


#     filename = f"{filename}_{D}D.txt" # prolly a good idea to add the dimension
#     if not cpp_utils.dataPath(filename).exists(): # If the file does not exist, run the gradient search code
#         total = len(Ns)*len(alphas)
#         time_start = time.time()
#         count = 0
#         for N in Ns:
#             for stepLength in stepLengths:
#                 for alpha in alphas:
#                     cpp_utils.gradientRun(logMet=logMet, lr=lr ,logEq=logEq, D=D, N=N, alpha=alpha, stepLength=stepLength, epsilon=epsilon, filename=filename, beta=beta, interacting=interacting)
#                     print(f"Gradient run done for alpha = {alpha} and N = {N}...")
#                     count += 1
#                     print(f"====>> Progress: {count}/{total} ({count/total*100:.2f}%)")
#                     print(f"====>>====>>Time elapsed: {time.time() - time_start:.2f} s")
#                     print(f"====>>====>>====>>Time remaining: {(time.time() - time_start)/count*(total-count)/60:.2f} min")


#     info = f"_D={D}_N={Ns}_stepLength={stepLengths[0]}_met={logMet}_eq={logEq}_eps={epsilon}"

#     df = cpp_utils.gradientLoad(filename=f"../Data/{filename}") # Load the data
#     df_detailed = cpp_utils.gradientLoad(filename=f"../Data/detailed_{filename}")

#     ax = sns.lineplot(data=df_detailed, x="Epoch", y="Alpha", hue="Particles", legend=True, palette=plot_utils.cmap)
#     ax.get_legend().remove()
#     plt.xlabel("Epoch")
#     plt.ylabel(r"$\alpha$")
#     # add legend
#     handles, labels = ax.get_legend_handles_labels()
#     ax.legend(handles=handles, labels=labels, title="Particles")


#     #norm = plt.Normalize(df_detailed["Alpha_0"].min(), df_detailed["Alpha_0"].max())
#     #sm = plt.cm.ScalarMappable(cmap=plot_utils.cmap, norm=norm)
#     #sm.set_array([])
#     #plt.colorbar(sm, label=r"$\alpha_0$", cmap=plot_utils.cmap)

#     if save:
#         plot_utils.save("alpha_search" + info)
#     plt.show()

#     return df, df_detailed, info


# def plot_energy_var(filename, df_detailed, info, save=False):
#     # Plot the energy variance

#     ax = sns.lineplot(data=df_detailed, x="Alpha", y=df_detailed["Energy_var"] / df_detailed["Particles"], hue="Particles", legend=True, palette=plot_utils.cmap)

#     #norm = plt.Normalize(df_detailed["Alpha_0"].min(), df_detailed["Alpha_0"].max())
#     #sm = plt.cm.ScalarMappable(cmap=plot_utils.cmap, norm=norm)
#     #sm.set_array([])
#     #plt.colorbar(sm, label=r"$\alpha_0$", cmap=plot_utils.cmap)

#     plt.xlabel(r"$\alpha$")
#     plt.ylabel(r"Var$(\langle E_L \rangle)/N$")


#     if save:
#         plot_utils.save(filename + info)
#     plt.show()


def plot_energy_per_epoch(filename="GD_energy_per_epoch", D=2, interacting=False, importance=False, save=False):
    Ns = np.array([2])
    
    lr = np.array([0.01, 0.001])
    logMet = 19 # 2 ^ 18 = 262144
    logEq = 14 # 10 % of logMet is 2^18 * 0.1 = 26214, while 2^15 = 32768

    # start time measurement
    start = time.time()



    
    for i, N in enumerate(Ns):
        info = f"N={N}_D={D}_met={logMet}_interacting={interacting}"

        filename += info
        if not cpp_utils.dataPath(filename + ".txt").exists():
        #total = len(Ns)*len(alphas)
            for j, l in enumerate(lr):
                print(f"Python Running for N = {N} and lr = {l}...")
                cpp_utils.nqsRun(D=D, N=N, logMet=logMet, logEq=logEq, stepLength=0.6, importance=importance, analytical=True, learnRate=l, interacting=interacting, filename=filename, detailed=False)
                print(f"Python run done for N = {N} and lr = {l}...")
    df_detailed = cpp_utils.nqsLoad(filename="detailed_" + filename+ ".txt")

    # remove column where lr = 1
    df_detailed = df_detailed[df_detailed.LearningRate != 0.01]

    # min energy value
    min_energy = df_detailed["Energy"].min()
    print(f"Minimum energy: {min_energy}")



    ax = sns.lineplot(data=df_detailed, x="Epoch", y="Energy", hue="LearningRate", legend=True, palette="pastel")
    ax.get_legend().remove()
    ax.set(xlabel=r"Epoch", ylabel=r"$\langle E_L \rangle [\hbar \omega]$")

    # add legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, title="Learning rate")

    # c = plot_utils.colors
    # fig, ax = plt.subplots()
    # for i, N in enumerate(Ns):
    #      df_detailed_N = df_detailed[ df_detailed.Particles == N ]
    #      E, E_std, epoch, MCCs = df_detailed_N["Energy"].to_numpy(), df_detailed_N["Energy_var"].to_numpy(), df_detailed_N["Epoch"].to_numpy(), df_detailed_N["Metro-steps"].to_numpy()
    #      ax.errorbar(epoch, E, np.sqrt(E_std)/(MCCs), c=c[i], label=f"{N =}")
    #      print(f"Minimum energy is {E.min():.4f} at epoch {epoch[E.argmin()]}")

    # ax.legend(ncol=4, bbox_to_anchor=(1.05, 1.15))
    # ax.set(xlabel=r"Epoch", ylabel=r"$\langle E_L \rangle [\hbar \omega]$")

    if save:
        plot_utils.save(filename.replace(".txt",f"_plot"))
    plt.show()

    # now plot the columns HiddenGradNorm     VisibleGradNorm      WeightGradNorm
    ax = sns.lineplot(data=df_detailed, x="Epoch", y="HiddenGradNorm", hue="LearningRate", legend=True, palette="pastel")
    ax.get_legend().remove()
    ax.set(xlabel=r"Epoch", ylabel=r"HiddenGradNorm")
    # add legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, title="Learning rate")
    plt.show()

    ax = sns.lineplot(data=df_detailed, x="Epoch", y="VisibleGradNorm", hue="LearningRate", legend=True, palette="pastel")
    ax.get_legend().remove()
    ax.set(xlabel=r"Epoch", ylabel=r"VisibleGradNorm")
    # add legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, title="Learning rate")
    plt.show()

    ax = sns.lineplot(data=df_detailed, x="Epoch", y="WeightGradNorm", hue="LearningRate", legend=True, palette="pastel")
    ax.get_legend().remove()
    ax.set(xlabel=r"Epoch", ylabel=r"WeightGradNorm")
    # add legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, title="Learning rate")
    plt.show()


if __name__ == "__main__":

    plot_energy_per_epoch(filename="GD_energy_per_epoch", D=2, interacting=True, importance=True, save=False)


    #df, df_detailed, info = plot_alpha_search(filename="alpha_search_ho_int",D=3, save=True, beta=1, interacting=True)

    #plot_energy_var("energy_var_alpha_search_ho_int", df_detailed, info, save=True)


    # plot alpha search for ho and eo in same plot
    #df_ho, df_detailed_ho, info_ho = plot_alpha_search(filename="alpha_search_ho_int",D=3, save=True, beta=1, interacting=True)
    #df_eo, df_detailed_eo, info_eo = plot_alpha_search(filename="alpha_search_eo_int",D=3, save=True, beta=1, interacting=True)

    # plot energy variance for ho and eo in same plot
    #plot_energy_var("energy_var_alpha_search_ho_int", df_detailed_ho, info_ho, save=True)
    #plot_energy_var("energy_var_alpha_search_eo_int", df_detailed_eo, info_eo, save=True)

