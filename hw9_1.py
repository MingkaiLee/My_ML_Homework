from hmmlearn import hmm
import numpy as np


if __name__ == '__main__':
    data = np.load('./HomeworkData/sequences.npy')
    length = np.ones(data.shape[0], dtype=np.int) * data.shape[1]
    data = data.reshape(-1, 1)
    # 隐状态
    states = [True, False]
    n_states = 2
    # 观测状态
    observations = range(1, 7)
    n_observations = 6
    # 建立模型
    model = hmm.MultinomialHMM(n_components=n_states, n_iter=3000, tol=0.001)
    # 模型训练问题
    model.fit(data, lengths=length)
    # 打印初始、转移、发射概率
    print('初始概率:')
    print(model.startprob_)
    print('转移概率:')
    print(model.transmat_)
    print('发射概率:')
    print(model.emissionprob_)
    # 模型预测问题
    ob_list = [3, 2, 1, 3, 4, 5, 6, 3, 1, 4, 1, 6, 6, 2, 6]
    ob_list = np.array(ob_list).reshape(-1, 1)
    logprob, action_list = model.decode(ob_list, algorithm='viterbi')
    print('该玩家的隐状态序列估计:')
    print(action_list)
