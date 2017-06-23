#include "importance_sampling.hpp"
#include "prior.hpp"
#include <iostream>
#include <ctime>


int main()
{
	double a1 = 10.747809151611289;
	double a2 = 0.36260141487530501;
	double a3 = 235000000000000.0;
	double a4 = 1.1587242790463443;
	double S = 0.1570168038792813;
	double z = 0.552631578947;
	TinkerPrior tp = TinkerPrior();

	std::cout.precision(17);
	std::cout << ImportanceSampling::fast_lognormal(22, 10, 30) << " = 0.00132871493565" << std::endl;
	std::cout << ImportanceSampling::p1(20, 19, 2) << " = 0.00997027749323" << std::endl;
	std::cout << ImportanceSampling::p2(1e4, 1e11, a1, a2, a3, a4, S, z) << " = 1.65626971034e-09" << std::endl;
	std::cout << ImportanceSampling::p3(1e11,z,tp) << " = 1.40762668993e-12" << std::endl;
	std::cout << ImportanceSampling::q1(20,19,2) << " = 0.00997027749323" << std::endl;
	std::cout << ImportanceSampling::q2(1e11, 1e4, a1, a2, a3, a4, S, z) << " = 9.18141342255e-51" << std::endl;

	static const double arr1[] = {13748.7935573,    8259.73850042 , 18382.3049543};
	std::vector<double> lum_obs (arr1, arr1 + sizeof(arr1) / sizeof(arr1[0]) );
	static const double arr2[] = {2.21, 2.02, 2.02};
	std::vector<double> zs (arr2, arr2 + sizeof(arr2) / sizeof(arr2[0]) );
	int nsamples = 100;
	
	// std::cout << tp.logpdf(2.2105263157900001, 691529487006.68616) << " f" << std::endl;

	// std::cout << "zs";
	// for(std::map<double, MassPrior>::iterator iter = tp.priormap.begin(); iter != tp.priormap.end(); ++iter)
	// {
		// double k =  iter->first;
		// std::cout << k << std::endl;
	// }
	std::cout << ImportanceSampling::log_multi_importance_sampling(lum_obs, zs, a1, a2, a3, a4, S, tp, nsamples) << " ~ -33.4533623069" << std::endl;

	clock_t begin = clock();
	ImportanceSampling::log_multi_importance_sampling(lum_obs, zs, a1, a2, a3, a4, S, tp, nsamples);
	clock_t end = clock();
	double timeSec = (end - begin) / static_cast<double>( CLOCKS_PER_SEC );
  	std::cout << timeSec;
	return 0;
}