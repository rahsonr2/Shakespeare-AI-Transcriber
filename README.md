# AI-Transcriber-CS630-Final (How to Run)

# 1(a). access either delta ai or purdue anvil via terminal
  Delta AI instructions:

  
  [delta_conda4.pdf](https://github.com/user-attachments/files/27166437/delta_conda4.pdf)

  Anvil instructions:

  
  [anvil_conda2.pdf](https://github.com/user-attachments/files/27166438/anvil_conda2.pdf)


# 1(b). upload file to super computer
  Delta AI:
  
  scp app.py (username)@dtai-login.delta.ncsa.illinois.edu

  Anvil:
  
  scp app.py (x-username)@anvil.rcac.purdue.edu:~

# 1(c). request resources
  Delta AI: 
  
  sacctmgr show associations user=$USER format=Account,User,Partition
  
  salloc --account=bgfm-dtai-gh --partition=ghx4 --gpus =1 --time=00:10:00
  
  ssh (gpu node given)



  Anvil: 

  salloc --partition=gpu --gres=gpu:1 --time=00:10:00

    
# 2. create a virtual environment
  module load python
  
  python3 -m venv my_venv1
  
  source my_venv1/bin/activate

# 3. install libraries
  pip install torch
  
  pip install streamlit
  
  pip install transformers
  
  pip install sentence_transformers

  
# 4. run script
  streamlit run app.py --server.address 0.0.0.0 --server.port 8501
