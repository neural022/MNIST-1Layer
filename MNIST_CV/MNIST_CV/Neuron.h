#pragma once

#include "stdlib.h"
#include "time.h"

#define PIXEL 784
class Neuron
{
	public:
		double x[PIXEL];
		double w[PIXEL];
		double y;
		double b;
		Neuron()
		{
			for (int i = 0; i < PIXEL; i++)
			{
				x[i] = 0;
				w[i] = rand()*1.0 / RAND_MAX;
			}
			y = b = 0;
		}
		~Neuron() {	}
};