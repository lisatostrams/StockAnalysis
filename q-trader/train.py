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
window_size = 60
agent = Agent(window_size)

#%%
offset = 10000
learn_size = 2000
data  = all_data[-learn_size-offset:-offset]
print(max(data) - min(data))


#%%
episode_count = 200
#data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 32
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
		reward = 0

		if action == 1 and len(agent.inventory) < 1: # buy
			buy_price = data[t] + (fee_mult * data[t])
			agent.inventory.append(buy_price)
			print( "Buy: " + formatPrice(data[t]))

		elif action == 2 and len(agent.inventory) > 0: # sell
			bought_price = agent.inventory.pop(0)
			cur_price = data[t]
			#cur_price = data[t] - (fee_mult * data[t])            
			#reward = cur_price - bought_price            
			reward = max(cur_price - bought_price, 0)
			total_profit += cur_price - bought_price
			print( 
                    "Sell: " + formatPrice(cur_price) + 
                    " | Profit: " + formatPrice(cur_price - bought_price),
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

	if e % 10 == 0:
		agent.model.save("/home/apeppels/q-trader/models/model_ep" + str(e))

#%%

x = np.arange(0,len(profits),1)
y = profits
plt.plot(x,y)