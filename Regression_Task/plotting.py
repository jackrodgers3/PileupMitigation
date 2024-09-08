import torch
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

predictions = torch.load(r'C:\Users\jackm\PycharmProjects\PileupMitigation\Regression_Task\predictions.pt')
truths = torch.load(r'C:\Users\jackm\PycharmProjects\PileupMitigation\Regression_Task\truths.pt')

predictions_flattened = []
truths_flattened = []
for i in range(len(predictions)):
    for j in range(len(predictions[i])):
        predictions_flattened.append(predictions[i][j])

for i in range(len(truths)):
    for j in range(len(truths[i])):
        truths_flattened.append(truths[i][j])

predictions_flattened = np.exp(np.array(predictions_flattened))
truths_flattened = np.exp(np.array(truths_flattened))

fig = plt.figure(figsize=(10, 8))
_, bins, _ = plt.hist(truths_flattened, histtype='step', bins=30, label='truth')
_ = plt.hist(predictions_flattened, histtype='step', bins=bins, label='pred')
plt.legend()
plt.yscale('log')
plt.savefig(r'C:\Users\jackm\PycharmProjects\PileupMitigation\Regression_Task\pt_comparison.png')
plt.close(fig)

