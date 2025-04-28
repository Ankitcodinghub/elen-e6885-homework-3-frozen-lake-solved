# elen-e6885-homework-3-frozen-lake-solved
**TO GET THIS SOLUTION VISIT:** [ELEN-E6885 Homework 3-Frozen Lake Solved](https://www.ankitcodinghub.com/product/elen-e6885-homework-frozen-lake-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;119801&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;1&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (1 vote)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;ELEN-E6885 Homework 3-Frozen Lake Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (1 vote)    </div>
    </div>
1 Introduction

1.1 Algorithms

The pseudo code for Policy Iteration, Value Iteration, Q-learning and SARSA:

(a) Policy Iteration (b) Value Iteration

(c) Q-Learning (d) SARSA

1.2 Implementation – OpenAI gym(FrozenLake-v1)

• Background

Winter is here. You and your friends were tossing around a frisbee at the park when you made a wild throw that left the frisbee out in the middle of the lake. The water is mostly frozen, but there are a few holes where the ice has melted. If you step into one of those holes, you’ll fall into the freezing water. At this time, there’s an international frisbee shortage, so it’s absolutely imperative that you navigate across the lake and retrieve the disc. However, the ice is slippery, so you won’t always move in the direction you intend. The episode ends when you reach the goal or fall in a hole [1].

• observation:

The states form a 4 * 4 grid. There are 4 kinds of states. “S” is the safe starting point, “F” represents frozen surface, which is safe as well. “H” represents a hole. You “fall to your doom” if you enter “H” states. “G” is your goal where the frisbee is located.

S F F F

F H F H

F F F H H F F G

• Actions:

Type: Discrete(4)

At each step, you can take 4 actions: “LEFT”, “DOWN”, “RIGHT”, “UP” represented by indices 0 to 3 respectively. Your next state is then given by the environment.

• Reward:

You receive a reward of 1 if you reach the goal, and 0 otherwise.

• Episode Termination:

The episode ends when you reach the goal or fall in a hole.

2 Environment Setup

We are using Google Colab [2] to execute the code. Colaboratory, or “Colab” for short, is a product from Google Research, allowing anybody to write and execute arbitrary python code through the browser, and is especially well suited to machine learning, data analysis and education. More technically, Colab is a hosted Jupyter notebook service that requires no setup to use, while providing free access to computing resources including GPUs.

Steps to use the environment with start code:

1. Upload the start code onto your Google Drive Account and open the FrozenLake.ipynbby using colab. (Right click on the file -&gt;open with -&gt;Google Colaboratory)

2. Mount the Google Drive onto the Colab as the storage location

Following the instructions returned from the below cell. You will need to click a web link and select the google account you want to mount (The one that you upload the start code), then copy the authorication code to the blank, press enter.

from google.colab import drive drive.mount(‘/content/gdrive’)

3. Append the directory location where you upload the start code folder to the sys.path

import sys

sys.path.append(‘/content/gdrive/My Drive/&lt;dir/to/start/code/folder&gt;’))

4. Then follow the detailed instructions step by step in FrozenLake.ipynb.

3 Task

You need to implement the following algorithms in RLalgs folder: Epsilon-Greedy, Policy Iteration, Value Iteration, Q-Learning and SARSA. Follow the detailed instructions step by step in FrozenLake.ipynb which interactively show the result of each algorithms. We also have gym-tutorial.ipynb for you to get familiar with Gym package.

4 Submission Instructions

Please submit the following two files:

1. A pdf report (printed out from Jupiter notebook with all codes and outcomes).

2. A zip file including folder ”RLalgs”, and your Jupyter notebook source file (for us tocheck your results).

References

[1] http://gym.openai.com/envs/FrozenLake-v0/

[2] https://colab.research.google.com/notebooks/intro.ipynb#recent=true
