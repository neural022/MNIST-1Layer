// MNIST_CV.cpp : 定義主控台應用程式的進入點。
//

#include "stdafx.h"
#include "stdlib.h"
#include "stdio.h"
#include "math.h"
#include "time.h"

#define TRAIN_NUM 60000
#define TEST_NUM 10000
#define PIXEL 784
#define NUMBER_NUM 10
#define LEARNING_RATE 0.001

class Neuron
{
	public:
		double x[PIXEL];
		double w[PIXEL];
		double y;

		Neuron() 
		{
			for (int i = 0; i < PIXEL; i++) x[i] = w[i] = 0;
			y = 0;
		}
		~Neuron() {	}
};

class MNIST
{
	private:
		char *trainFile;
		char *trainlabelFile;
		char *testFile;
		char *testlabelFile;
		FILE *trainf;
		FILE *trainlabelf;
		FILE *testf;
		FILE *testlabelf;
		Neuron neuron[NUMBER_NUM];
		double yy[NUMBER_NUM];
	public:
		MNIST(int fNameLength)
		{
			trainFile = new char[fNameLength];
			trainlabelFile = new char[fNameLength];
			testFile = new char[fNameLength];
			testlabelFile = new char[fNameLength];
		}
		~MNIST()
		{
			delete testFile;
			delete testlabelFile;
			delete trainlabelFile;
			delete trainFile;
		}

		void printImage()
		{
			for (int j = 0; j < NUMBER_NUM / NUMBER_NUM; j++)
			{
				for (int k = 0; k < PIXEL; k++)
				{
					if (k % 28 == 0) printf("\n");
					if(this->neuron[j].x[k] != 0) printf("*");
					else printf(" ");
				}
				printf("\n");
			}
		}

		void readImage(char c, FILE *f)
		{
			for (int j = 0; j < PIXEL; j++)
			{
				double x = 0;
				fscanf(f, "%c", &c);
				if ((double)c) x = 1;
				for (int k = 0; k < NUMBER_NUM; k++) this->neuron[k].x[j] = x;
			}
			//	printImage();
		}

		void readLabel(char c, FILE *f)
		{
			fscanf(f, "%c", &c);
			for (int i = 0; i < NUMBER_NUM; i++) this->yy[i] = 0;
			this->yy[(int)c] = 1;
		}
		
		double getErr(double yy, double y) { return yy - y; }

		void updateWeights(int n)
		{
			for (int i = 0; i < PIXEL; i++) this->neuron[n].w[i] += LEARNING_RATE * this->neuron[n].x[i] * getErr(this->yy[n], this->neuron[n].y);	// *(this->neuron[n].y * (1 - this->neuron[n].y));
		}

		double sigmoid(double t) { return 1 / (1 + exp(-1.0 * t)); }

		void f(int n, int m)
		{
			double sum = 0;
			for (int i = 0; i < n; i++) sum += this->neuron[m].w[i] * this->neuron[m].x[i];
			this->neuron[m].y = sigmoid(sum);
		}

		void train()
		{
			for (int i = 0; i < NUMBER_NUM; i++)
			{
				f(PIXEL, i);
				updateWeights(i);
			}
		}

		void test() { for (int i = 0; i < NUMBER_NUM; i++) f(PIXEL, i); }

		int predict(Neuron *neuron)
		{
			double max = 0;
			int number = 0;
			for (int i = 0; i < NUMBER_NUM; i++)
			{
				if (neuron[i].y > max)
				{
					max = neuron[i].y;
					number = i;
				}
			}
			return number;
		}

		int getLabel() { for (int i = 0; i < NUMBER_NUM; i++) if (this->yy[i] == 1) return i; }

		void readTrainData(char trainFileName[], char trainlabelFileName[])
		{
			int correct = 0;
			char c, s;
			trainf = fopen(trainFileName, "rb");
			trainlabelf = fopen(trainlabelFileName, "rb");
				//	Read Info
				for (int i = 0; i < 16; i++) fscanf(trainf, "%c", &c);
				for (int i = 0; i < 8; i++) fscanf(trainlabelf, "%c", &s);
				//	Read Pixel
				for (int i = 0; i < TRAIN_NUM; i++)
				{
					readImage(c, trainf);
					readLabel(s, trainlabelf);
					train();
					int predict_num = predict(this->neuron);
					if (predict_num == getLabel()) correct++;
					//	printf("Predict: %d\tAnswer: %d\n", predict_num, getLabel());
				}
				double persentage = (double)(correct) / TRAIN_NUM;
				printf("accuracy:\t%d%%\n", (int)(persentage * 100));
			fclose(trainlabelf);
			fclose(trainf);
		}

		void readTestData(char testFileName[], char testlabelFileName[])
		{
			int correct = 0;
			char c, s;
			testf = fopen(testFileName, "rb");
			testlabelf = fopen(testlabelFileName, "rb");
				//	Read Info
				for (int i = 0; i < 16; i++) fscanf(testf, "%c", &c);
				for (int i = 0; i < 8; i++) fscanf(testlabelf, "%c", &s);
				//	Read Pixel
				for (int i = 0; i < TEST_NUM; i++)
				{
					readImage(c, testf);
					readLabel(s, testlabelf);
					test();
					int predict_num = predict(this->neuron);
					if (predict_num == getLabel()) correct++;
					//	printf("Predict: %d\tAnswer: %d\n", predict_num, getLabel());
				}
				double persentage = (double)(correct) / TEST_NUM;
				printf("accuracy:\t%d%%\n", (int)(persentage*100));
			fclose(testlabelf);
			fclose(testf);
		}
};

int main()
{
	MNIST mnist(30);
	printf("training ...\n");
	mnist.readTrainData("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
	printf("\ntesting ...\n");
	mnist.readTestData("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
}



