# 将原本的样例进行sclice扩充
import logging
import numpy as np


def slice_data( data_x, data_y, slice_ratio): 
    n = data_x.shape[0] # data_x的样例数量
    length = data_x.shape[1] # 时间步长度
    n_dim = data_x.shape[2] # for MTS  维度
    nb_classes = data_y.shape[1] # 类别种类数量

    length_sliced = int(length * slice_ratio) #新的样例的（被slice之后）的时间步长度
    
    # 样例翻倍数
    increase_num = length - length_sliced + 1 #if increase_num =5, it means one ori becomes 5 new instances.
    # 新的样例总数 = 旧的样例总数 * 翻倍数
    n_sliced = n * increase_num

    new_x = np.zeros((n_sliced, length_sliced,n_dim))
    new_y = np.zeros((n_sliced,nb_classes))
    for i in range(n):#对原本的每一例
        for j in range(increase_num): #原本一例对应的当前例
            # 移动length_sliced长度的窗口 每次移动一个单位
            new_x[i * increase_num + j, :,:] = data_x[i,j : j + length_sliced,:]
            new_y[i * increase_num + j] = np.int_(data_y[i].astype(np.float32))

    return new_x, new_y

# 没用到
def split_train( train_x,train_y):
    #shuffle for splitting train set and dataset
    n = train_x.shape[0]
    ind = np.arange(n)
    np.random.shuffle(ind) #shuffle the train set
    
    #split train set into train set and validation set
    valid_x = train_x[ind[0:int(0.2 * n)]]
    valid_y = train_y[ind[0:int(0.2 * n)]]

    ind = np.delete(ind, (range(0,int(0.2 * n))))

    train_x = train_x[ind] 
    train_y = train_y[ind]
        
    return train_x,train_y,valid_x,valid_y

def _movingavrg(  data_x, window_size):
    num = data_x.shape[0] # 样例个数
    length_x = data_x.shape[1] # 时间步长度
    num_dim = data_x.shape[2] # for MTS  维度
    output_len = length_x - window_size + 1 # 输出数据 的时间步长度
    output = np.zeros((num, output_len,num_dim))
    for i in range(output_len): #对于每一个时间步 进行产生
        output[:,i] = np.mean(data_x[:, i : i + window_size], axis = 1) # 所有第i个时间步 = 原本的i~i+window_size 的值的平均
    return output

# 进行num次 moving average 每一次的移动窗口大小从windowbase，windowbase+stepsize, windowbase+2step_size...
# 将num次的moving average结果拼接在一起
# 数据的个数没有改变，长度改变了
def movingavrg(  data_x, window_base, step_size, num):
    if num == 0:
        return (None, [])
    out =  _movingavrg(data_x, window_base)
    data_lengths = [out.shape[1]] # 当前处理得到的数据的 时间步长度
    for i in range(1, num): # 执行num次 每次window_size 增加step_size
        window_size = window_base + step_size * i
        if window_size > data_x.shape[1]:
            continue
        new_series =  _movingavrg(data_x, window_size)
        data_lengths.append( new_series.shape[1])
        out = np.concatenate([out, new_series], axis = 1)
    return (out, data_lengths)

def batch_movingavrg(  train,valid,test, window_base, step_size, num):
    (new_train, lengths) =  movingavrg(train, window_base, step_size, num)
    (new_valid, lengths) =  movingavrg(valid, window_base, step_size, num)
    (new_test, lengths) =  movingavrg(test, window_base, step_size, num)
    return (new_train, new_valid, new_test, lengths)

def _downsample(  data_x, sample_rate, offset = 0):
    num = data_x.shape[0]
    length_x = data_x.shape[1]
    num_dim = data_x.shape[2] # for MTS 
    last_one = 0
    if length_x % sample_rate > offset:
        last_one = 1
    new_length = int(np.floor( length_x / sample_rate)) + last_one
    output = np.zeros((num, new_length,num_dim))
    for i in range(new_length):
        output[:,i] = np.array(data_x[:,offset + sample_rate * i])

    return output

def downsample(  data_x, base, step_size, num):
    # the case for dataset JapaneseVowels MTS
    if data_x.shape[1] ==26 :
        return (None,[]) # too short to apply downsampling
    if num == 0:
        return (None, [])
    out = _downsample(data_x, base,0)
    data_lengths = [out.shape[1]]
    #for offset in range(1,base): #for the base case
    #    new_series = _downsample(data_x, base, offset)
    #    data_lengths.append( new_series.shape[1] )
    #    out = np.concatenate( [out, new_series], axis = 1)
    for i in range(1, num):
        sample_rate = base + step_size * i 
        if sample_rate > data_x.shape[1]:
            continue
        for offset in range(0,1):#sample_rate):
            new_series =  _downsample(data_x, sample_rate, offset)
            data_lengths.append( new_series.shape[1] )
            out = np.concatenate( [out, new_series], axis = 1)
    return (out, data_lengths)

def batch_downsample(  train,valid,test, window_base, step_size, num):
    (new_train, lengths) =  downsample(train, window_base, step_size, num)
    (new_valid, lengths) =  downsample(valid, window_base, step_size, num)
    (new_test, lengths) =  downsample(test, window_base, step_size, num)
    return (new_train, new_valid, new_test, lengths)



def augumentation(x_train, valid, window_base, step_size):
    slice_ratio = 0.9
    ori_len = x_train.shape[1]  
    if ori_len > 500 : 
        slice_ratio = slice_ratio if slice_ratio > 0.98 else 0.98
    elif ori_len < 16:
        slice_ratio = 0.7
        
    increase_num = ori_len - int(ori_len * slice_ratio) + 1 #this can be used as the bath size
    logging.info(f"increase_num(数据量翻倍数): {increase_num}")
    # n_train_batch number of train batches 是批次的数量，根据批次数量确定批次大小
    # train_batch_size 批次大小
    train_batch_size = int(x_train.shape[0] * increase_num / n_train_batch)
    if train_batch_size > max_train_batch_size : 
        # limit the train_batch_size 
        n_train_batch = int(x_train.shape[0] * increase_num / max_train_batch_size)
    
    logging.info(f"train_batch_size(批次的大小): , {train_batch_size},n_train_batch(批次的数量): {n_train_batch}")
    # data augmentation by slicing the length of the series 
    x_train,y_train = self.slice_data(x_train,y_train,slice_ratio)
    x_val,y_val = self.slice_data(x_val,y_val,slice_ratio)
    x_test,y_test = self.slice_data(x_test,y_test,slice_ratio)

    train_set_x, train_set_y = x_train,y_train
    valid_set_x, valid_set_y = x_val,y_val
    test_set_x, _ = x_test,y_test

    logging.info(f"---After slicing---\n train_set_x.shape:{train_set_x.shape} train_set_y.shape:{train_set_y.shape}, valid_set_x.shape:{valid_set_x.shape} valid_set_y.shape:{valid_set_y.shape}") 
    # 验证集的数量
    valid_num = valid_set_x.shape[0]
     # logging.info("increase factor is ", increase_num, ', ori len', ori_len)
    # 原本的验证集数量 作为batch数量
    valid_num_batch = int(valid_num / increase_num)

    test_num = test_set_x.shape[0]
    # 原本的测试集数量 作为batch数量
    test_num_batch = int(test_num / increase_num)

    #slicing之后的时间步长
    length_train = train_set_x.shape[1] #length after slicing
    
    # # 根据window_size 比例设定window大小
    # window_size = int(length_train * window_size) if window_size < 1 else int(window_size)
    # # 进行Moving Average和Downsample的window大小，必须是window_size的整数倍，且不能大于length_train
    # #*******set up the ma and ds********#
    
    
    # # 最多可以进行多少次降采样操作
    # # 通过除以 (pool_factor * window_size)，它确保每次降采样操作后，剩余的序列长度仍然足够进行后续的池化操作。
    # ds_num_max = length_train / (pool_factor * window_size)
    # logging.info(f"ds_num_max:{ds_num_max}, length_train:{length_train}, pool_factor:{pool_factor}, window_size:{window_size}")
    # ds_num = int(min(ds_num, ds_num_max))
    
    logging.info(f"Before moving average and downsample:\n the data length is {length_train}\n ma_base(移动窗口大小):{ma_base} ma_step(移动窗口每次的增加值):{ma_step} ma_num:{ma_num}\n ds_base:{ds_base} ds_step:{ds_step} ds_num:{ds_num}")
    #*******set up the ma and ds********#

    (ma_train, ma_valid, ma_test , ma_lengths) = batch_movingavrg(train_set_x,
                                                    valid_set_x, test_set_x,
                                                    ma_base, ma_step, ma_num)
    (ds_train, ds_valid, ds_test , ds_lengths) = batch_downsample(train_set_x,
                                                    valid_set_x, test_set_x,
                                                    ds_base, ds_step, ds_num)
    logging.info("---moving average and downsample DONE----")
    logging.info(f"ma_train.shape = {ma_train.shape},ma_lengths = {ma_lengths}")
    logging.info(f"ds_train.shape = {ds_train.shape},ds_lengths = {ds_lengths}")
    #concatenate directly
    data_lengths = [length_train] 
    #downsample part:
    if ds_lengths != []:
        data_lengths +=  ds_lengths
        train_set_x = np.concatenate([train_set_x, ds_train], axis = 1)
        valid_set_x = np.concatenate([valid_set_x, ds_valid], axis = 1)
        test_set_x = np.concatenate([test_set_x, ds_test], axis = 1)

    #moving average part
    if ma_lengths != []:
        data_lengths += ma_lengths
        train_set_x = np.concatenate([train_set_x, ma_train], axis = 1)
        valid_set_x = np.concatenate([valid_set_x, ma_valid], axis = 1)
        test_set_x = np.concatenate([test_set_x, ma_test], axis = 1)
    # logging.info("Data length:", data_lengths)
    
    logging.info(f"After ma and ds , train_set_x.shape:{train_set_x.shape},valid_set_x.shape:{valid_set_x.shape}, data_lengths:{data_lengths}")
    
    n_train_size = train_set_x.shape[0]
    n_valid_size = valid_set_x.shape[0]
    n_test_size = test_set_x.shape[0]
    batch_size = int(n_train_size / n_train_batch)
    n_train_batches = int(n_train_size / batch_size)
    data_dim = train_set_x.shape[1]  
    num_dim = train_set_x.shape[2] # For MTS 
    nb_classes = train_set_y.shape[1] 

    logging.info(f"batch size:{batch_size}" )
    logging.info(f'n_train_batches is {n_train_batches}')
    logging.info(f'train size(样例个数):{n_train_size}, valid size:{n_valid_size} test size:{n_test_size}')
    logging.info(f'data dim is(时间步长) :{data_dim}')
    logging.info(f'num dim is(特征数量) :{num_dim}')
    logging.info('---------------------------')

    ######################
    # BUILD ACTUAL MODEL #
    ######################