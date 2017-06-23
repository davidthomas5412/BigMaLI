#ifndef PRIOR_H
#define PRIOR_H

#include <vector>
#include <map>
#include <random>

class MassPrior
{
	public:
		MassPrior(std::vector<double>& masses, std::vector<double>& probs);
		double pdf(const double mass);
		double logpdf(const double mass);
		void rvs(std::vector<double>& samples);
		void print();
	private:
		std::vector<double> prob;
		std::vector<double> mass;
		std::vector<double> cumsum;
		std::default_random_engine gen;
    	std::uniform_real_distribution<double> dist;
		double trapz(const std::vector<double>& x, const std::vector<double>& y);
};

class TinkerPrior
{
	public:
		TinkerPrior();
		/*
		Might need to be vector of z's
		*/
		double pdf(const double z, const double mass);
		double logpdf(const double z, const double mass);
		void rvs(const double z, std::vector<double>& samples);
		MassPrior fetch(const double z);
	private:
        static long snap(const double z);
		std::map<long, MassPrior> priormap;
};

#endif