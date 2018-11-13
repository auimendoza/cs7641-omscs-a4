required versions:
  python      3.6.6 
  numpy       1.15.1
  pandas      0.23.4
  matplotlib  2.2.3
  seaborn     0.9.0

1. Install OpenAI Gym:
   $ pip install gym

2. Install pymdptoolbox:
   $ pip install pymdptoolbox

3. Run Policy Iteration
   $ for i in 0 2
     do
     python pi.py $i
     done

4. Run Value Iteration
   $ for i in 0 2
     do
     python vi.py $i
     done

5. Run Q Learning
   $ for i in 0 2
     do
     python q.py $i
     done

EOF   
