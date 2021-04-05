# QMethodUtilities

# Instructions for Simulation

It has changed a bit from the last commit. Most importantly, the fill simulation and the flush analysis are in different modules.
The main function is in program/src/Qsimulation.cpp. It reads the config json file and constructs a simulator object and a few analyzer objects. Check the code to see how the config json file is read. 
The simulator class is in SimModule, and it is unique. The analyzer classes are in AnalysisModule, and they have to be registered to the simulator module. The simulator generates a batch of flushes, and then each analyzer will do analysis of these flushes. 
If you want to write your own analysis module, follow these instructions.
1: Define a class inheritated from AnalysisModuleBase.
2: Override the functions FlushAnalysis for flush-by-flush analysis, EndAnalysis for operations after all flushes are simulated, and Output for writing data out.
3: The Simulator will pass the device addresses of the flush histograms to the FlushAnalysis. You need to write the GPU kernel for your own analysis using these address for data source.
4: Add the header for your analyzer in AnalysisModuleList.h
5: Add how to register your anlayzer in SimModule/src/AnalyzerRegistration.cu
6: Add the rule for compiling your .cu file in the Makefile, and link the .o file to the executable.
7: Add The configuration to the input json file.

You have examples to follow when you pull the respository. I have already implemented my own analyzer class. 
