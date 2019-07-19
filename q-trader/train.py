from agent.agent import Agent
from functions import *
import sys
import matplotlib.pyplot as plt

#if len(sys.argv) != 4:
#	print("Usage: python train.py [stock] [window] [episodes]")
#	exit()

#stock_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
#%%
all_data = [float(x.split(',')[6]) for x in open('../in_a_minute_ta.csv').read().splitlines()[78:]]
#%%
window_size = 500
agent = Agent(window_size)

#%%
# how far back to sample the learning data
offset = 10000
# size of the episode
learn_size = 1000
data  = all_data[-learn_size-offset:-offset]
# how much can we make with the best single buy-and-hold?
print(max(data) - min(data))


#%%
episode_count = 300
#data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 2
fee_mult = 0 #0.00075 / 100.
profits = []

for e in range(episode_count + 1):
	print("Episode " + str(e) + "/" + str(episode_count))
	state = getState(data, 0, window_size + 1)

	total_profit = 0
	agent.inventory = []

	for t in range(l):
		action = agent.act(state)

		# sit
		next_state = getState(data, t + 1, window_size + 1)
		impatience = 0.001   
		action_reward = 0.1
		reward = 0 - impatience
		        
		max_bets = 1
		if action == 1 and len(agent.inventory) < max_bets: # buy
			buy_price = data[t] + (fee_mult * data[t])
			agent.inventory.append(buy_price)
			reward = action_reward
			print( "Buy: " + formatPrice(data[t]))

		elif action == 2 and len(agent.inventory) > 0: # sell
			bought_price = agent.inventory.pop(0)
			cur_price = data[t]
			#cur_price = data[t] - (fee_mult * data[t])
            
			#set this limit to a nonzero number to allow for negative reward            
			max_punishment = 0.1
			diff = cur_price - bought_price
			reward = max(diff, -max_punishment) - impatience
			total_profit += diff
			print( 
                    "Sell: " + formatPrice(cur_price) + 
                    " | Profit: " + formatPrice(diff),
                    " | Reward: " + str(reward),
                    " | Index: " + str(t),
                    " | Epsilon: " + str(agent.epsilon)
                                 
                                 )

		done = True if t == l - 1 else False
		agent.memory.append((state, action, reward, next_state, done))
		state = next_state

		if done:
			print( "--------------------------------")
			print( "Total Profit: " + formatPrice(total_profit))
			print( "--------------------------------")
			profits.append(total_profit)
		memlen = len(agent.memory)
		if memlen > batch_size:
			#print("Replaying " + str(memlen))
			agent.expReplay(batch_size)
	plot_profit(profits)
	if e % 3 == 0:
		agent.model.save("/home/apeppels/StockAnalysis/q-trader/models/model_ep" + str(e))

#%%

def plot_profit(y):
    x = np.arange(0,len(y),1)
    plt.plot(x,y)
    
plot_profit(profits)