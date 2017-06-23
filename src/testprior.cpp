#include "prior.hpp"
#include <iostream>
#include <vector>
#include <cmath>

int main()
{
	TinkerPrior tp = TinkerPrior();
	double z = 0;
	
	//test sampling
	int n = pow(10, 6);
	std::vector<double> samples(n);
	tp.rvs(z, samples);
	double mean = 0;
	for (int i=0; i<n; i++)
	{
		mean += samples[i];
	}
	std::cout << (mean / n) << std::endl;
	// test evaluation
	std::cout.precision(17);
	static const double tmp[] = {12565562999.1,
		37101091908.6,
		20276524417.6,
		12760674392.2,
		19789997964.2,
		13937924000.6,
		15650179359.0,
		23178136425.8,
		40223016628.0,
		25603331470.3
	};
	std::vector<double> masses(tmp, tmp + sizeof(tmp) / sizeof(tmp[0]) );

	std::vector<double> solution = std::vector<double>(10);

	
	tp.pdf(z, masses, solution);

	std::cout << "solution: " << std::endl;
	for (int i=0; i<10; i++)
	{
		std::cout << solution[i] << std::endl;
	}

	return 0;
}