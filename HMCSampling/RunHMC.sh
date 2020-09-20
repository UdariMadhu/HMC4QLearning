# python3 RunHMC.py --sdim 3 --srange 0 1 0 1 0 1 --stepsize 0.02 --trlen 100 --burn 10 --hmcsample 100 --cSig 80 --HMCseed 123 --scov 0.025 0.06 0.09  --mean 0 0 0 

python3 -u  -W ignore RunHMC.py --sdim 4 --srange -0.785 0.785 -3 3 -2.4 2.4 -3.5 3.5 --ssize 5 --scov 15 15 15 15 --samples 2 --stepsize 0.02 --trlen 100 --burn 0 --hmcsample 20 --cSig 80