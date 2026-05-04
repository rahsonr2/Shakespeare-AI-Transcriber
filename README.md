# AI-Transcriber-CS630-Final (How to Run)

# 1(a). Access either Delta AI or Purdue Anvil via Terminal
  Delta AI instructions:

  
  [delta_conda4.pdf](https://github.com/user-attachments/files/27166437/delta_conda4.pdf)

  Purdue Anvil instructions:

  
  [anvil_conda2.pdf](https://github.com/user-attachments/files/27166438/anvil_conda2.pdf)


# 1(b). Upload File to Supercomputer
  Delta AI:
  
    scp app.py (username)@dtai-login.delta.ncsa.illinois.edu

  Anvil:
  
    scp app.py (x-username)@anvil.rcac.purdue.edu:~

# 1(c). Request Resources
  Delta AI: 
  
    sacctmgr show associations user=$USER format=Account,User,Partition
    
    salloc --account=bgfm-dtai-gh --partition=ghx4 --gpus =1 --time=00:10:00
  
    ssh (gpu node given)



  Anvil: 

    salloc --partition=gpu --gres=gpu:1 --time=00:10:00

    
# 2. Create a Virtual Environment
    module load python
  
    python3 -m venv my_venv1
  
    source my_venv1/bin/activate

# 3. Install Python Libraries
  
    pip install torch
  
    pip install streamlit
  
    pip install transformers
  
    pip install sentence_transformers

    pip install groq

  
# 4(a). Run Script (Supercomputer)


  (Local PC (PowerShell))
  
    ssh -L 8501:(node):8501 (username)@delta.ncsa.illinois.edu
  
  **Note: If it asks for a password, it is (your access password) + (6 digit code from duo app). Ex: Password123?012345**

  
  (Delta/Anvil) 
  
    streamlit run app.py --server.address 0.0.0.0 --server.port 8501
    
  Run in browser to open web app
    
    http://localhost:8501 


# 4(b). Run Script (Local PC)
  
  Open project in Visual Studio Code
  
  Install libraries listed above in terminal

      streamlit run app.py

  Web app will automatically open if ran on local PC
    

    
        

  
