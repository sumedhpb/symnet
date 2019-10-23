import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

df_adaptive=pd.read_csv("./SymNet_Rahul/symnet/tests/BostonHousing/method_8/training_adaptive.log")
df=pd.read_csv("./SymNet_Rahul/symnet/tests/BostonHousing/method_8/training_0.1.log")

#Plot loss
loss=df.loc[:,'mean_absolute_error'].values
loss_adaptive=df_adaptive.loc[:,'mean_absolute_error'].values
plt.plot(np.arange(len(loss)),loss,color='b')
plt.axhline(y=0.358, color='r', linestyle='-')
plt.plot(np.arange(len(loss_adaptive)),loss_adaptive,color='k')
plt.xlabel("Iteration")
plt.ylabel("loss")
plt.title("Loss over time")
plt.legend(['Loss-Constant LR=0.1','y=0.358','Loss-Adaptive LR'])
plt.savefig("./SymNet_Rahul/symnet/tests/BostonHousing/method_8/loss_compare.png")

val_loss=df.loc[:,'val_mean_absolute_error'].values
val_loss_adaptive=df_adaptive.loc[:,'val_mean_absolute_error'].values
plt.figure()
plt.plot(np.arange(len(val_loss)),val_loss,color='b')
plt.axhline(y=0.337, color='r', linestyle='-')
plt.plot(np.arange(len(val_loss_adaptive)),val_loss_adaptive,color='k')
plt.xlabel("Iteration")
plt.ylabel("Validation loss")
plt.title("Validation loss over time")
plt.legend(['Loss-Constant LR=0.1','y=0.337','Loss-Adaptive LR'])
plt.savefig("./SymNet_Rahul/symnet/tests/BostonHousing/method_8/val_loss_compare.png")


#Plot learning rate
plt.figure()
lr=df.loc[:,'lr'].values
lr_adaptive=df_adaptive.loc[:,'lr'].values
plt.plot(np.arange(len(lr)),lr,color='b')
plt.plot(np.arange(len(lr_adaptive)),lr_adaptive,color='r')
plt.xlabel("Iteration")
plt.ylabel("Learning rate")
plt.title("Learning rate over time")
plt.legend(['LR=0.1','Adaptive LR'])
plt.savefig("./SymNet_Rahul/symnet/tests/BostonHousing/method_8/lr_compare.png")
