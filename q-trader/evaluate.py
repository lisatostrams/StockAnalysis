import keras
from keras.models import load_model

from agent.agent import Agent
from functions import *

model_name = 'model_ep100'
model = load_model("models/" + model_name)
window_size = model.layers[0].input.shape.as_list()[1]

agent = Agent(window_size, True, model_name)
#data = getStockDataVec(stock_name)
offset = 20000
data = all_data[-5000 - offset:-offset]
l = len(data) - 1

state = getState(data, 0, window_size + 1)
total_profit = 0
agent.inventory = []

for t in range(l):
	action = agent.act(state)

	# sit
	next_state = getState(data, t + 1, window_size + 1)
	reward = 0

	if action == 1 and len(agent.inventory) < 1: # buy
		agent.inventory.append(data[t])
		print( "Buy: " + formatPrice(data[t]))

	elif action == 2 and len(agent.inventory) > 0: # sell
		bought_price = agent.inventory.pop(0)
		reward = max(data[t] - bought_price, 0)
		total_profit += data[t] - bought_price
		print(
			"Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price),
			"Balance: " + str(total_profit)			
			)

	done = True if t == l - 1 else False
	agent.memory.append((state, action, reward, next_state, done))
	state = next_state

	if done:
		print("--------------------------------")
		print(" Total Profit: " + formatPrice(total_profit))
		print("--------------------------------")
