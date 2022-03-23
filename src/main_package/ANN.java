package main_package;

import java.io.File;
import java.io.FileNotFoundException;
import java.text.DecimalFormat;
import java.util.Random;
import java.util.Scanner;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.TransferFunctionType;

public class ANN{
	private static File file;
	NeuralNetwork<BackPropagation> networkBP;
	NeuralNetwork<BackPropagation> networkMBP;
	
	DataSet trainDatasetBP;
	DataSet testDatasetBP;
	DataSet trainDatasetMBP;
	DataSet testDatasetMBP;
	
	BackPropagation backPropogation;
	MomentumBackpropagation momentumBackPropogation;
	
	double[] totalErrorBP, maxVals, minVals;
	
	
	public ANN() throws FileNotFoundException {
		file = new File("Data.txt");
		maxVals = new double[4];
		minVals = new double[4];
		
		findMaxVals(file);
	}
	
	public void annBP(int epochBP, int inputLayerBP, int outputLayerBP, int hiddenLayerBP) throws FileNotFoundException{
		totalErrorBP = new double[(int)epochBP];
		
		networkBP =  new MultiLayerPerceptron(TransferFunctionType.SIGMOID, inputLayerBP, hiddenLayerBP, outputLayerBP);
		trainDatasetBP = new DataSet(inputLayerBP, outputLayerBP);
		testDatasetBP = new DataSet(inputLayerBP, outputLayerBP);
		
		backPropogation = new BackPropagation();
		
		setDataSets();
	}
	
	public void annMBP(int inputLayerMBP, int outputLayerMBP, int hiddenLayerMBP) throws FileNotFoundException {
		networkMBP = new MultiLayerPerceptron(TransferFunctionType.SIGMOID, inputLayerMBP, hiddenLayerMBP, outputLayerMBP);
		trainDatasetMBP = new DataSet(inputLayerMBP, outputLayerMBP);
		testDatasetMBP = new DataSet(inputLayerMBP, outputLayerMBP);
		
		momentumBackPropogation = new MomentumBackpropagation();
		
		setDataSets();
	}

	public void initializeNetworBP(int epochBP,  double learningRateBP) {
		backPropogation.setMaxIterations(epochBP);
		backPropogation.setLearningRate(learningRateBP);
		networkBP.setLearningRule(backPropogation);
	}
	
	public void initializeNetworkMBP(int epochMBP, double learningRateMBP, double momentumMBP) {
		momentumBackPropogation.setMaxIterations(epochMBP);
		momentumBackPropogation.setLearningRate(learningRateMBP);
		momentumBackPropogation.setMomentum(momentumMBP);
		networkMBP.setLearningRule(momentumBackPropogation);
	}
		
	private void setDataSets() throws FileNotFoundException {
		Scanner sc = new Scanner(file);
		int index_train= 0, index_test = 0;
		double[] input_train, output_train, input_test, output_test;
		Random rand = new Random();
						
		while(sc.hasNext()) {
			if(rand.nextInt(1000000000)%3 == 0 && index_test < 300) {
				index_test++;
				input_test = new double[] {normalize(sc.nextDouble(), maxVals[0], minVals[0]), normalize(sc.nextDouble(), maxVals[1], minVals[1]), normalize(sc.nextDouble(), maxVals[2], minVals[2])};
				output_test= new double[] {normalize(sc.nextDouble(), maxVals[3], minVals[3])};
				
				DataSetRow test_row = new DataSetRow(input_test, output_test);
				testDatasetBP.add(test_row);
				testDatasetMBP.add(test_row);				
			}
			else if(index_train < 700) {
				index_train++; 
				input_train = new double[] {normalize(sc.nextDouble(), maxVals[0], minVals[0]), normalize(sc.nextDouble(), maxVals[1], minVals[1]), normalize(sc.nextDouble(), maxVals[2], minVals[2])};
				output_train = new double[] {normalize(sc.nextDouble(), maxVals[3], minVals[3])};

				DataSetRow train_row = new DataSetRow(input_train, output_train);
				trainDatasetBP.add(train_row);
				trainDatasetMBP.add(train_row);
			}
		}
		sc.close();
	}
	
	private double test(NeuralNetwork network, DataSet testDataSet, String savedNetworkName) {
		network = NeuralNetwork.createFromFile(savedNetworkName);
		double totalError = 0;
		for(DataSetRow row : testDataSet) {
			network.setInput(row.getInput());
			network.calculate();
			totalError += meanSquaredError(row.getDesiredOutput(), network.getOutput());
		}
		 return totalError/testDataSet.size();
	}
	
	private double meanSquaredError(double[] desired, double[] result) {
		double oneRowError = 0;
		for(int i=0; i<desired.length; i++) {
			oneRowError += Math.pow(desired[i] - result[i], 2);
		}
		return oneRowError/desired.length;
	}
	
	private void findMaxVals(File file) throws FileNotFoundException {
		Scanner in = new Scanner(file);
		
		for(int i=0; i<4; i++) {
			maxVals[i] = Double.MIN_VALUE;
			minVals[i] = Double.MAX_VALUE;
		}
		while(in.hasNextDouble()) {
			for(int i=0; i<4; i++) {
				double data = in.nextDouble();
				if(data > maxVals[i]) maxVals[i] = data;
				if(data < minVals[i]) minVals[i] = data;
			}
		}
		in.close();
	}
	
	private double normalize(double x, double xMin, double xMax) {
		return (x-xMin)/(xMax-xMin);
	}
	
	public void trainBP(int epoch) {
		for(int i=0; i<epoch; i++) {
			networkBP.getLearningRule().doOneLearningIteration(trainDatasetBP);
			if(i==0) totalErrorBP[i] = 1;
			else totalErrorBP[i] = networkBP.getLearningRule().getPreviousEpochError();
		}
		networkBP.save("backPropogation.nnet");
		//System.out.println("Back Propogation Training was Done!");
	}
	
	public void trainMBP() {
		networkMBP.learn(trainDatasetMBP);
		networkMBP.save("momentumBackPropogation.nnet");
		//System.out.println("Momentum Back Propogation Training was Done!");
	}
	
	public void testforBP() {
		DecimalFormat df = new DecimalFormat("###.##########");
		System.out.println("Test Error Back Propagation:" + df.format(test(networkBP, testDatasetBP, "backPropogation.nnet")));
	}
	public void testForMBP() {
		DecimalFormat df = new DecimalFormat("###.##########");
		System.out.println("Test Error Momentum Back Propagation:" + df.format(test(networkMBP, testDatasetMBP, "momentumBackPropogation.nnet")));
	}
	
	public void totalErrorBP() {
		DecimalFormat df = new DecimalFormat("###.##########");
		System.out.println("Error Back Propagation     :" + df.format(backPropogation.getTotalNetworkError()));	
	}
	
	public void eachEpochErrorBP() {
		DecimalFormat df = new DecimalFormat("###.##########");
		System.out.println("\nEach Epoch Errors in Back Propogation\n-----------------------------------------");
		for(double er : totalErrorBP) {
			//System.out.print(++index +" ");
			System.out.println(df.format(er));
		}
	}
	
	public void totalErrorMBP() {
		DecimalFormat df = new DecimalFormat("###.##########");
		System.out.println("Error Momentum Back Propagation     :" + df.format(momentumBackPropogation.getTotalNetworkError()));
	}
}