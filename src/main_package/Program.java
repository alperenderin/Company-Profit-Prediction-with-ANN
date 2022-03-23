package main_package;

import java.io.FileNotFoundException;
import java.util.Scanner;

public class Program {
	
	public static void main(String[] args) throws FileNotFoundException {
		//////////////////////////////////////////////////////////////
		int epochBP = 300,  hiddenLayerBP = 10,  inputLayerBP = 3,  outputBP = 1,  
			epochMBP = 100, hiddenLayerMBP = 10, inputLayerMBP = 3, outputMBP = 1; 
		double learningRateBP = 0.2, learningRateMBP = 0.2, momentumMBP = 0.8;
		
		int user_choice = 0, choice = 0;
		boolean loop = true;
		//////////////////////////////////////////////////////////////
		ANN ann = new ANN();
		
		Scanner input = new Scanner(System.in);
		do {
			System.out.print("1-Backpropagation\n2-Backpropagation with Momentum\n0-Exit Program\n: ");
			user_choice = input.nextInt();
			
			switch(user_choice) {
			case 1:
				System.out.printf("------------ Current Parameters ------------\n"
								  + "Input Layers:%d Hidden Layers:%d Output Layers:%d\n"
						          + "Epoch:%d Learning Rate:%f\n"
						          + "-------------------------------------------\n", inputLayerBP, hiddenLayerBP, outputBP, epochBP, learningRateBP);
				System.out.print("Hidden Layer: ");
				hiddenLayerBP = input.nextInt();
				System.out.print("Epoch: ");
				epochBP = input.nextInt();
				System.out.print("Learning Rate: ");
				learningRateBP = input.nextDouble();
				
				ann.annBP(epochBP, inputLayerBP, inputLayerBP, hiddenLayerBP);
				ann.initializeNetworBP(epochBP, learningRateBP);
				
				System.out.println("\n------------Back Propagation-------------");
				ann.trainBP(epochBP);
				
				ann.totalErrorBP();
				ann.testforBP();
				System.out.print("------------------------------------------------------");

				do {
					System.out.println("Do you want to each epoch errors?\n(0-No 1-Yes)");
					choice = input.nextInt();
					
					if(choice == 1)
						ann.eachEpochErrorBP();
				}while(choice != 0);			
				break;
			case 2:
				System.out.printf("------------ Current Parameters ------------\n"
						  + "Input Layers:%d Hidden Layers:%d Output Layers:%d\n"
				          + "Epoch:%d Learning Rate:%f Momentum:%f\n"
				          + "-------------------------------------------", inputLayerMBP, hiddenLayerMBP, outputMBP, epochMBP, learningRateMBP, momentumMBP);
				System.out.print("Hidden Layer: ");
				hiddenLayerMBP = input.nextInt();
				System.out.print("Epoch: ");
				epochMBP = input.nextInt();
				System.out.print("Learning Rate: ");
				learningRateMBP = input.nextDouble();
				
				ann.annMBP(inputLayerMBP, inputLayerMBP, hiddenLayerMBP);
				ann.initializeNetworkMBP(epochMBP, learningRateMBP, momentumMBP);

				System.out.println("------------Momentum Back Propagation-------------");
				ann.trainMBP();
				
				ann.totalErrorMBP();
				ann.testForMBP();
				System.out.print("------------------------------------------------------");
				break;
			case 0:
				loop = false;
				System.out.println("Exiting Program");
				break;
			default:
				System.out.print("Please provide correct input value!\n(1-No Momentum 2-With Momentum 0-Exit Program): ");
				user_choice = input.nextInt();
				if(user_choice == 0) {
					loop = false;
					System.out.println("Exiting Program");
				}
			}
		}while(loop == true);
				
		
		
		
	}
}
