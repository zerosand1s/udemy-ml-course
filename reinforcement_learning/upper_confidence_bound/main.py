import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

dataset = pd.read_csv('Ads_CTR_Optimization.csv')

total_rounds = dataset.shape[0]
number_of_ads = dataset.shape[1]
number_of_selections = [0] * number_of_ads
sums_of_rewards = [0] * number_of_ads
ads_selected = []
total_reward = 0

for round in range(0, total_rounds):
  ad = 0
  max_upper_bound = 0
  for i in range(0, number_of_ads):
    if number_of_selections[i] > 0:
      average_reward = sums_of_rewards[i] / number_of_selections[i]
      delta_i = math.sqrt(3/2 * (math.log(round + 1) / number_of_selections[i]))
      upper_bound = average_reward + delta_i
    else:
      upper_bound = 1e400
    if upper_bound > max_upper_bound:
      max_upper_bound = upper_bound
      ad = i

  ads_selected.append(ad)
  number_of_selections[ad] = number_of_selections[ad] + 1
  reward = dataset.values[round, ad]
  sums_of_rewards[ad] = sums_of_rewards[ad] + reward
  total_reward = total_reward + reward


print(total_reward)

plt.hist(ads_selected)
plt.title('Histogram of Ad Selections')
plt.xlabel('Ads')
plt.ylabel('Number of Ad Selections')
plt.show()