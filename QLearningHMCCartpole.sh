python3 -u QLearningHMCCartpole.py --sdim 4 --adim 1 --srange -0.785 0.785 -3 3 -2.4 2.4 -3.5 3.5 --arange -10 10 --ssize 5 --asize 5 --scov 10 10 10 10 --max-bias 10 --samples 200 --steps 100 --seed 32 --stepsize 0.02 --trlen 100 --burn 0 --hmcsample 10 --cSig 80 --mode iid

# python3 QLearningHMCCartpole.py --sdim 4 --adim 1 --srange -0.785 0.785 -3 3 -2.4 2.4 -3.5 3.5 --arange -10 10 --ssize 5 --asize 5 --scov 5 5 5 5 --max-bias 10 --samples 300 --steps 500 --seed 32
