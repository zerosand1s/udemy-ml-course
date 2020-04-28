import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random

dataset = pd.read_csv('Ads_CTR_Optimization.csv')

total_rounds = dataset.shape[0]
number_of_ads = dataset.shape[1]
number_of_rewards_1 = [0] * number_of_ads
number_of_rewards_0 = [0] * number_of_ads
ads_selected = []
total_reward = 0

for round in range(0, total_rounds):
  ad = 0
  max_random_draw = 0
  for i in range(0, number_of_ads):
    random_beta = random.betavariate(number_of_rewards_1[i] + 1, number_of_rewards_0[i] + 1)
    if random_beta > max_random_draw:
      max_random_draw = random_beta
      ad = i

  ads_selected.append(ad)
  reward = dataset.values[round, ad]
  if reward == 1:
    number_of_rewards_1[ad] = number_of_rewards_1[ad] + 1
  else:
    number_of_rewards_0[ad] = number_of_rewards_0[ad] + 1
  total_reward = total_reward + reward


print(total_reward)

plt.hist(ads_selected)
plt.title('Histogram of Ad Selections')
plt.xlabel('Ads')
plt.ylabel('Number of Ad Selections')
plt.show()