// MNIST_CV.cpp : 定義主控台應用程式的進入點。
//

#include "stdafx.h"
#include "stdlib.h"
#include "stdio.h"
#include "math.h"
#include "time.h"
#include "Neuron.h"

#define TRAIN_NUM 60000
#define TEST_NUM 10000
#define NUMBER_NUM 10
//	#define LEARNING_RATE 0.01

class MNIST
{
	private:
		unsigned char c, s;
		FILE *trainf, *trainlabelf;
		FILE *testf, *testlabelf;
		Neuron neuron[NUMBER_NUM];
		double yy[NUMBER_NUM];
		double LEARNING_RATE;
	public:
		MNIST(char trainFileName[], char trainlabelFileName[], char testFileName[], char testlabelFileName[])
		{
			LEARNING_RATE = 0.5;
			trainf = fopen(trainFileName, "rb");
			trainlabelf = fopen(trainlabelFileName, "rb");
			testf = fopen(testFileName, "rb");
			testlabelf = fopen(testlabelFileName, "rb");
		}
		~MNIST()
		{
			fclose(trainlabelf);
			fclose(trainf);
			fclose(testlabelf);
			fclose(testf);
		}

		void printImage()
		{
			for (int i = 0; i < NUMBER_NUM / NUMBER_NUM; i++)
			{
				for (int j = 0; j < PIXEL; j++)
				{
					if (j % 28 == 0) printf("\n");
					if(this->neuron[i].x[j] != 0) printf("*");
					else printf(" ");
				}
				printf("\n");
			}
		}

		void readInfo(unsigned char c, unsigned char s, FILE *f, FILE *g)
		{
			for (int i = 0; i < 16; i++) fscanf(f, "%c", &c);
			for (int i = 0; i < 8; i++) fscanf(g, "%c", &s);
		}

		void readData(unsigned char c, unsigned char s, FILE *f, FILE *g)
		{
			//	Read Pixel
			for (int i = 0; i < PIXEL; i++)
			{
				double x = 0;
				fscanf(f, "%c", &c);
				if ((double)c) x = 1;
				for (int j = 0; j < NUMBER_NUM; j++) this->neuron[j].x[i] = double(c) / 255;
			}
			//	printImage();
			//	Read Label
			fscanf(g, "%c", &s);
			for (int i = 0; i < NUMBER_NUM; i++) this->yy[i] = 0;
			this->yy[(int)s] = 1;
		}
		
		double sigmoid(double t) { return 1 / (1 + exp(-1.0 * t)); }

		void f(int n, int m)
		{
			double sum = 0;
			for (int i = 0; i < n; i++) sum += this->neuron[m].w[i] * this->neuron[m].x[i];
			this->neuron[m].y = sigmoid(sum + this->neuron[m].b);
		}

		double getErr(double yy, double y) { return yy - y; }

		void updateWeights(int n)
		{
			for (int i = 0; i < PIXEL; i++) this->neuron[n].w[i] += this->LEARNING_RATE * this->neuron[n].x[i] * getErr(this->yy[n], this->neuron[n].y);// *(this->neuron[n].y) * (1 - this->neuron[n].y);
			this->neuron[n].b += this->LEARNING_RATE * getErr(this->yy[n], this->neuron[n].y);
		}

		int predict(Neuron *neuron)
		{
			double max = 0;	int number = 0;
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

		void printInfo(int *correct, int n)
		{
			int predict_num = predict(this->neuron);
			if (predict_num == getLabel()) (*correct)++;
			printf("N: %5d\tPredict: %d\tAnswer: %d\t", n + 1, predict_num, getLabel());
			double persent = (double)(*correct) / (n + 1);
			printf("current accuracy: %lf\n", persent);
		}

		void train()
		{
			int correct = 0;
			readInfo(c, s, trainf, trainlabelf);
			for (int i = 0; i < TRAIN_NUM; i++)
			{
				readData(c, s, trainf, trainlabelf);
				for (int j = 0; j < NUMBER_NUM; j++)
				{
					f(PIXEL, j);
					updateWeights(j);
				}
				if (i % 15000 == 0) this->LEARNING_RATE /= 5;
				printInfo(&correct, i);
			}
		}

		void test()
		{
			int correct = 0;
			readInfo(c, s, testf, testlabelf);
			for (int i = 0; i < TEST_NUM; i++)
			{
				readData(c, s, testf, testlabelf);
				for (int j = 0; j < NUMBER_NUM; j++) f(PIXEL, j);
				if (i % 15000 == 0) this->LEARNING_RATE /= 5;
				printInfo(&correct, i);
			}
		}
};

int main()
{
	srand(time(NULL)); rand();
	MNIST mnist("train-images.idx3-ubyte", "train-labels.idx1-ubyte", "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
	printf("training ...\n");
	mnist.train();
	printf("------------------train ending----------------\n");
	printf("press enter to testing ...\n");
	getchar();
	mnist.test();
	printf("\ncost execute time: %.2lfseconds\n", (double)clock() / CLOCKS_PER_SEC);

	return 0;
}



