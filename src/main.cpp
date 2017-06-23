#include <iostream>
#include <vector>
#include "prior.hpp"
#include "importance_sampling.hpp"
#include "hyperparameters.hpp"
#include <algorithm>
#include <fstream>
#include <ctime>
#include <random>

int main()
{

	/**
	test showing benefit of using raw arrays
	... will swith after testing
	**/
	

	clock_t begin = clock();
	const int nsteps = 3e3;
	const int nsamples = 100;
	TinkerPrior tp = TinkerPrior();
	std::ifstream infile("mock_data.txt");
	const int entries = 115919;
	std::vector<double> zs(entries);
	std::vector<double> lum_obs(entries);
	double z, lum_ob;
	int i = 0;
	while (infile >> z >> lum_ob)
	{
		zs[i] = z;
		lum_obs[i] = lum_ob;
		i += 1;
	}
  	
  	std::default_random_engine gen;
  	std::normal_distribution<double> dist_a1(10.709, 0.022);
  	std::normal_distribution<double> dist_a2(0.359, 0.009);
  	std::normal_distribution<double> dist_a4(1.10, 0.06);
  	std::normal_distribution<double> dist_S(0.155, 0.0009);
	double a1, a2, a4, S, ans;
	const double a3 = 2.35e14;
	std::cout.precision(17);
	for (int steps=0; steps<nsteps; steps++)
	{
		a1 = dist_a1(gen);
		a2 = dist_a2(gen);
		a4 = dist_a4(gen);
		S = dist_S(gen);
		ans = ImportanceSampling::log_multi_importance_sampling(lum_obs, zs, a1, a2, a3, a4, S, tp, nsamples);
		std::cout << a1 << " " << a2 << " " << a4 << " " << S << " " << ans << std::endl;
	}
	clock_t end = clock();
	double timeSec = (end - begin) / static_cast<double>( CLOCKS_PER_SEC );
  	std::cout << timeSec;
}