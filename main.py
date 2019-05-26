#coding=utf-8
import os
import numpy as np
from U_net_convlstm2d import get_unet
from dice_loss import dice_coef_loss

#读取数据，可以在此函数中划分训练集、验证集和测试集
def load_data(data_path):
    samples=os.listdir(data_path)
    all_samples_input=[]
    all_samples_target=[]
    for sample in samples:
        temp=np.load(os.path.join(data_path,sample))
        all_samples_input.append(temp['input'])
        all_samples_target.append(temp['target'])
    return np.array(all_samples_input),np.array(all_samples_target)

def main():
    inputs,targets=load_data('data')
    model=get_unet(input_size=(None,64,64,3))
    # model.compile(loss=dice_coef_loss, optimizer='adadelta')
    model.fit(inputs,targets,batch_size=1,epochs=10)
if __name__ == "__main__":
    main()