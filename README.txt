Code Repository: https://github.com/weiguanghuang/cs7641

Requirements:
ABAGAIL: https://github.com/pushkar/ABAGAIL
Apache Ant: https://ant.apache.org/bindownload.cgi
https://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html
Configure Java and Ant to windows environment and path variables. This could be a useful guide if needed https://www.mkyong.com/ant/how-to-install-apache-ant-on-windows/

Download and Build ABAGAIL jar package
cd ABAGAIL
ant
java -cp ABAGAIL.jar opt.test.XORTest
java -cp ABAGAIL.jar opt.test.TravelingSalesmanTest
reference here: https://github.com/pushkar/ABAGAIL/wiki

Part 1
Neural Network Weight Optimization

	Dataset:
	phishing.csv
	Original Phishing Websites Data - available at https://www.openml.org/d/4534

	Edit the dataset path in phishing.java (in particular in the BufferedReader in the method initializeInstances()) to match the downloading directory

	In the source code directory Compile from command prompt
	javac phishing.java

	run 
	java -cp ABAGAIL.jar rand_opt.phishing

	output will be collected in folder Optimization_Results

Part 2
More Optimization Problems
	Under the source code directory, compile from command prompt
	javac Opt_ContinuousPeaks.java
	javac Opt_FourPeaks.java
	javac Opt_TravelingSalesman.java

	run 
	java -cp ABAGAIL.jar rand_opt.Opt_ContinuousPeaks
	java -cp ABAGAIL.jar rand_opt.Opt_FourPeaks
	java -cp ABAGAIL.jar rand_opt.Opt_TravelingSalesman

	output will be collected in folder Optimization_Results


