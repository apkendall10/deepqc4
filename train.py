import agent4 as a4, gameControls4 as gc, time, numpy as np, pandas as pd
from matplotlib import pyplot as plt 

fpath1 = 'a1'
fpath2 = 'a2'

start_time = time.time()

mya = a4.agent(fpath=None, debug=False, lr=.01, full_layer = False, use_target = False)
mya2 = a4.agent(fpath=None, debug=False, lr=.01, use_target = False, advanced_memory = True, full_layer = False)

for j in range(10):
    print('round', j)
    score = []
    loss1 = []
    loss2 = []
    playCount = []

    #mya = a4.agent(fpath=None, debug=False, lr=.01, advanced_memory = False, use_target = True)
    #mya2 = a4.agent(fpath=None, debug=False, lr=.01, advanced_memory = True, use_target = True, cnn = True)

    for i in range(100):
        p1Wins, p2Wins, ties, l1, l2, pCount = gc.pick_winner(mya,mya2, season = True, rounds = 100)
        val = [p1Wins, p2Wins, ties]
        loss1.append(l1)
        loss2.append(l2)
        score.append(val)
        playCount.append(pCount)
        print(i,round((time.time() - start_time)/60),val,round(l1,3),round(l2,3),pCount)
        details = gc.play_game(1,mya,mya2, debug=True)
        mya.update_target(1)
        mya2.update_target(1)

    score = np.array(score)
    print(score[:,0].mean(),score[:,1].mean())

    x = np.arange(len(loss1))
    y1 = score[:,0]
    y2 = score[:,1]
    z1 = np.array(loss1) * 100
    z2 = np.array(loss2) * 100
    if False:
        plt.scatter(x,y1, label = 'A1 Wins')
        plt.scatter(x,y2, label = 'A2 Wins')
        plt.scatter(x,z1, label = 'A1 Loss')
        plt.scatter(x,z2, label = 'A2 Loss')
        plt.legend(loc = 'best')
        plt.xlabel("Epoch")
        plt.show()

    playCount = np.array(playCount)

    data = np.array([y1,y2,z1,z2, playCount])
    df = pd.DataFrame(data.T, columns= ['A1 Wins', 'A2 Wins', 'A1 Loss', 'A2 Loss', 'Play Count'])
    fnam = 'data/memory_lr_opp3/test{}.txt'.format(j+1)
    df.to_csv(fnam, index = False)

mya.save_agent(fpath1)
mya2.save_agent(fpath2)