To install the environment follow following steps.

1. Install Minconda/Anaconda in your system
2. Open "Minconda/Anaconda prompt shell" on windows and "terminal" on linux systems.
3. Run "conda create --name <your env name> --file <path to dla2.txt>"
4. Modify only Pipeline/changerollno.py
5. Test experiments by running "python3 main.py T/F" (T for using GPU and F for using CPU)
6. Write code for saving checkpoints in "trainer" function.