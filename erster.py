#!/usr/bin/env python3



import numpy as np
#import bisect
import torch as pt
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import List, Tuple

from SVD import *
from functions import *
# make results reproducible
pt.manual_seed(0)

# increase plot resolution
plt.rcParams["figure.dpi"] = 160

# create output directory
output = "output"
#!mkdir -p $output


pt.autograd.set_detect_anomaly(True)

##################################################################


# define the name of the directory to be created
model_save = "/home/jan/FirstNN/training/"

# define datasets 
# training data sind die ersten 200 Zeitschritte mit 10 Koeffizienten [batch,Anzahl Coeff]

train_data = pt.zeros([200,10])				
for i in range (len(train_data)-1):
	for n in range (0,10):
		train_data[i , n] = modeCoeff[i , n]
		
val_data = pt.zeros([10,10])
for i in range (0,10):
	for n in range (0,10):
		val_data[i , n] = modeCoeff[i+200, n]

test_data = pt.zeros([20,10])
for i in range (0,len(test_data)):
	for n in range (0,10):
		test_data[i , n] = modeCoeff[i+220 , n]

#assert pt.isclose(train_data_norm.min(), pt.tensor(-1.0))
#assert pt.isclose(train_data_norm.max(), pt.tensor(1.0))
#assert pt.allclose(train_data, scaler.rescale(train_data_norm))
	
	
#print("train_data", train_data.shape, "modeCoeffshape", modeCoeff.shape, "val_data", val_data.shape, "test_data", test_data.shape)

#######################################################################################  
 
model_params = {
    "n_inputs": 10,
    "n_outputs": 10,
    "n_layers": 4,
    "n_neurons": 20,
    "activation": pt.nn.ReLU()	#fast and accurate
}
learnRate=0.001
epochs=10000


model = FirstNN(**model_params)
print()
print(model)


#######################################################################################
#######################################################################################


train_loss, val_loss = optimize_model(model, train_data,val_data, epochs, lr=learnRate, save_best=model_save)


#######################################################################################
fig, ax = plt.subplots()
epochs = len(train_loss)                                      
plt.plot(range(1, epochs+1), train_loss, lw=1.0, label="training loss")
plt.plot(range(1, epochs+1), val_loss, lw=1.0, label="validation loss")
plt.xlabel("epoch")
plt.ylabel("MSE loss")
plt.xlim(1, epochs+1)
plt.yscale("log")
plt.legend()
#plt.show()
plt.savefig(f"{model_save}loss.png")


#######################################################################################
# test best model
#######################################################################################
#predict just the next timestep, then use the correct one and predict the next

best_model = FirstNN(**model_params)
best_model.load_state_dict(pt.load(f"{model_save}best_model_train.pt"))
pred = pt.zeros([len(test_data),10])
pred3 = pt.zeros(10)
for i in range (0, len(test_data)):
	prediction = best_model(test_data[i]).squeeze()
	pred[i] = prediction
#	print("pred3",pred3.shape,pred3)
#print("	pred   ", pred)
pred_print = pred.detach().numpy()

#######################################################################################
#prediction with predicted values
#eingabe ist letzte Test_data
pred4 = pt.zeros([len(test_data),10])
pred4[0] = pred[len(test_data)-1]

for i in range (1, len(test_data)):
	prediction = best_model(pred4[i-1]).squeeze()
	pred3 = prediction
	pred4[i] = pred3[:10]
#	print("pred3",pred3.shape,pred3)
	
		
#print("	pred4   ", pred4)
pred4_print = pred4.detach().numpy()
#######################################################################################
times_num = [float(time) for time in window_times]
x = []
x2 = []
for i in range (0 , len(test_data)):
	x.append(times_num[i]+5.50)
	x2.append(times_num[i]+5.50+0.025*len(test_data))
fig, axarr = plt.subplots(2, 2, sharex=True, sharey=True)
count = 0
for row in range(2):
    for col in range(2):
        axarr[row, col].plot(times_num, modeCoeff[:,count], lw=1, label=f"coeff. mode {i+1}")
        axarr[row, col].plot(x, pred_print[:,count], lw=1, label=f"coeff. mode {i+1}")
        axarr[row, col].plot(x2, pred4_print[:,count], lw=1, label=f"coeff. mode {i+1}")
        axarr[row, col].set_xlim(min(times_num), max(times_num)+1.00)
        axarr[row, col].grid()
        # add 1 for the POD mode number since we subtracted the mean
        axarr[row, col].set_title(f"mode coeff. {count + 1}")
        count += 1
for ax in axarr[1, :]:
    ax.set_xlabel(r"$t$ in $s$")
#plt.show()
plt.savefig(f"{model_save}prediction.png")

