import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PPG_file = '/home/v-jiewang/ContentANDStyle_Disentangle/ppg_spk_norm/assets_25_10ms/PPGs_VCTK105/p225/p225_002.npy'
PPG_value = np.load(PPG_file)
sns.heatmap(PPG_value, cmap='Reds')
plt.savefig('./ppg_0.png')  # [-0.2, 0.8]
print(PPG_value.shape)
# print(PPG_value)


# txt_for_check = './spmel_p226_010.txt'
# for ppg in PPG_value:
#     f = open(txt_for_check, 'a')
#     f.write('\n' + str(ppg))
#     f.close()


PPG_downsample_file = '/home/v-jiewang/ContentANDStyle_Disentangle/ppg_spk_norm/assets_64_16ms/PPGs_VCTK105_64_16_nonorm_notrim/p225/p225_002.npy'
PPG_downsample = np.load(PPG_downsample_file)
sns.heatmap(PPG_downsample, cmpa='Reds')
plt.savefig('./ppg_1.png')  # [-0.2, 0.8]
print(PPG_downsample.shape)
# print(PPG_value)
