python3 -u  -W ignore QLearningHMCCartpole.py --sdim 4 --adim 1 --srange -0.785 0.785 -1.5 1.5 -1.2 1.2 -1.5 1.5 --arange -10 10 --ssize 10 --asize 100 --scov 15 15 15 15 --max-bias 10 --samples 3000 --steps 350 --stepsize 0.02 --trlen 100 --burn 0 --hmcsample 20 --cSig 80 --mode hmc

# python3 QLearningHMCCartpole.py --sdim 4 --adim 1 --srange -0.785 0.785 -3 3 -2.4 2.4 -3.5 3.5 --arange -10 10 --ssize 5 --asize 5 --scov 5 5 5 5 --max-bias 10 --samples 300 --steps 500 --seed 32
