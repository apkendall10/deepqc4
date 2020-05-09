import pandas as pd, numpy as np, math, random, time, agent4 as a4, board4
from itertools import permutations 

#Game Type
#0: agent vs. agent
#1: agent vs. random
#2: agent vs. player
def play_game(game_type, agent1, agent2 = None, gamma = .95, debug = False):
    board = board4.board()
    start_player = random.randrange(-1,2,2)
    player = start_player
    move_choice = -1
    p1_b = -1
    p1_m = 0
    p2_b = -1
    p2_m = 0
    memory = []
    memory2 = []
    while(not board.game_over()):

        if player == 1:
            #update previous state info
            if p1_b != -1:
                p1_nb = board.copy()
                done = False
                reward = 0
                memory.append([p1_b,p1_m,reward,p1_nb,done])
            move_choice = agent1.pick_move(board,player)[0]
            p1_b = board.copy()
            p1_m = move_choice
        else:
            if p2_b != -1:
                p2_nb = board.copy()
                done = False
                reward = 0
                memory2.append([p2_b,p2_m,reward,p2_nb,done])
            
            if game_type == 0: 
                move_choice = agent2.pick_move(board,player)[0]
            
            elif game_type == 1:
                move_choice = random.choice(board.available_columns)
            
            else:
                move_choice = -1
                while not move_choice in board.available_columns:
                    move_choice = int(input('Select move (0-6)'))
            
            p2_b = board.copy()
            p2_m = move_choice
        board.do_move(move_choice, player)
        player = player * -1
        if game_type in [2]:
            board.show_board()
    outcome = board.check_win()
    done = True
    memory.append([p1_b,p1_m,outcome,p1_nb,done])
    memory2.append([p2_b,p2_m,outcome,p2_nb,done])
    if debug:
        board.show_board()

    for mems in [memory, memory2]:
            num_plays = len(mems)
            for play in range(num_plays):
                mems[play][2] = math.pow(gamma, num_plays - play - 1) * outcome
    
    if True:
        temp1 = []
        temp2 = []
        temps = [temp1, temp2]
        memories = [memory2, memory]
        for i in range(len(temps)):
            temp = temps[i]
            mems = memories[i]
            for play in range(len(mems)):
                b, move, reward, next_b, done = mems[play]
                temp.append([b.switch(),move,reward*-1, next_b.switch(), done])
        memory += temp1
        memory2 += temp2
    return [outcome, memory, memory2]

#have two agents compete for a number of rounds and pick the network that performs better
def pick_winner(agent1, agent2, season = False, rounds = 250, use_memory = True):
    p1_wins = 0
    p2_wins = 0
    l1 = []
    l2 = []
    play_count = []
    for i in range(rounds):
        game_data = play_game(0,agent1,agent2, debug = False)
        outcome, game_log1, game_log2 = game_data
        if outcome == 1:
            p1_wins +=1
        elif outcome == -1:
            p2_wins +=1

        play_count.append(len(game_log1))

        l1.append(agent1.learn(game_log1, 1, use_memory = use_memory))
        l2.append(agent2.learn(game_log2, -1, use_memory = use_memory))

    p1_success = p1_wins > p2_wins
    if season:
        return [p1_wins, p2_wins, rounds - p1_wins - p2_wins, np.array(l1).mean(),
         np.array(l2).mean(), np.array(play_count).mean()]
    if p1_wins > p2_wins:
        return agent1
    else:
        return agent2

#run a tournament of agents to find the best one
def run_tournament(agents,initial_learning_rate = .1):
    rounds = len(agents)/2
    if rounds == 1:
        print(rounds)
        return pick_winner(agents[0],agents[1], math.pow(initial_learning_rate,rounds))
    elif rounds < 1:
        return agents[0]
    else:
        half = int(len(agents)/2)
        first_half = run_tournament(agents[0:half])
        second_half = run_tournament(agents[half:len(agents)])
        print(rounds)
        return pick_winner(first_half,second_half, math.pow(initial_learning_rate, rounds))
    #do some stuff

#run a season of agent competition with an end-season tournament
def run_season(agents, tournament_size, debug = False):
    teams = np.arange(len(agents))
    scores = dict(zip(teams,np.zeros(len(agents))))
    games = list(permutations(teams, 2))
    random.shuffle(games)
    for cycle in range(2):
        for game in games:
            home_team = game[0]
            away_team = game[1]
            #print('teams', home_team, away_team)
            result = pick_winner(agents[home_team],agents[away_team],True, debug = debug)
            scores[home_team] += result[0]
            scores[away_team] += result[1]
            agents[home_team] = result[2]
            agents[away_team] = result[3]
        print(scores)
    standings = sorted(scores.items(), key = lambda x: x[1], reverse = True)
    top_teams = [int(x[0]) for x in standings[0:tournament_size]]
    print(top_teams)
    tournament_teams = []
    home_team = 0
    away_team = len(top_teams)-1
    while(home_team < away_team):
        tournament_teams.append(agents[home_team])
        tournament_teams.append(agents[away_team])
        home_team += 1
        away_team -= 1
    return run_tournament(tournament_teams,.01)