import agent as a4, gameControls as gc, time, numpy as np, pandas as pd, sys, os

fpath1 = 'a1'
fpath2 = 'a2'

rounds = 1
if len(sys.argv) > 1:
    rounds = int(sys.argv[1])

start_time = time.time()

path = 'data/train'
error = True
fileNum = 0
while error and fileNum < 5000:
    fileNum += 1
    try:
        os.mkdir(path + str(fileNum))
        error = False
        print('Writing to ' + path + str(fileNum))
    except:
        pass


mya = a4.agent(fpath=fpath1, debug=False, lr=.01, full_layer = False, use_target = True)
mya2 = a4.agent(fpath=fpath2, debug=False, lr=.01, use_target = True, advanced_memory = True, full_layer = False)

for j in range(rounds):
    print('round', j)
    score = []
    loss1 = []
    loss2 = []
    playCount = []

    #mya = a4.agent(fpath=None, debug=False, lr=.01, advanced_memory = False, use_target = True)
    #mya2 = a4.agent(fpath=None, debug=False, lr=.01, advanced_memory = True, use_target = True, cnn = True)

    for i in range(50):
        p1Wins, p2Wins, ties, l1, l2, pCount = gc.pick_winner(mya,mya2, season = True, rounds = 50)
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

    playCount = np.array(playCount)

    mya.save_agent(fpath1)
    mya2.save_agent(fpath2)

    data = np.array([y1,y2,z1,z2, playCount])
    df = pd.DataFrame(data.T, columns= ['A1 Wins', 'A2 Wins', 'A1 Loss', 'A2 Loss', 'Play Count'])
    fnam = path + str(fileNum) + '/test{}.txt'.format(j+1)
    df.to_csv(fnam, index = False)

    