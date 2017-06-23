#include "prior.hpp"
#include <algorithm>
#include <iostream>
#include <cmath>
#include <fstream>

double MassPrior::trapz(const std::vector<double>& x, const std::vector<double>& y)
{
	double sum, avg_height, width;
	sum = 0;
	for (int i=0; i < x.size()-1; i++)
	{
		avg_height = (0.5 * (y[i] + y[i+1]));
		width = x[i+1] - x[i];
		sum = sum + width * avg_height;
	}
	return sum;
}

MassPrior::MassPrior(std::vector<double>& masses, std::vector<double>& probs)
{
	const double epsilon = 1e-30;
	std::vector<double> mass(410);
	std::vector<double> prob(410);
	mass = masses;
	prob = probs;

	mass.insert(mass.begin(), mass[0] - 1);
	mass.insert(mass.begin(), 1);
	mass.push_back(mass.back() + 1);
	mass.push_back(mass.back() * 100);

	prob.insert(prob.begin(), epsilon);
	prob.insert(prob.begin(), epsilon);
	prob.push_back(epsilon);
	prob.push_back(epsilon);


	const int n = prob.size();
	const double norm_trapz = trapz(mass, prob);
	double norm_sum = 0;
	std::vector<double> cumsum = std::vector<double>(n);
	for (int i=0; i<n; i++)
	{
		prob[i] = prob[i] / norm_trapz;
		norm_sum += prob[i];
	}
	cumsum[0] = 0;
	for (int i=1; i<n; i++)
	{
		cumsum[i] = cumsum[i-1] + (prob[i] / norm_sum);
	}

	this->mass = mass;
	this->prob = prob;
	this->cumsum = cumsum;

    std::uniform_real_distribution<double> dist(0, 1);
    this->dist = dist;
}

double MassPrior::pdf(const double mass)
{
	auto upper = std::upper_bound(this->mass.begin(), this->mass.end(), mass);
	int right_ind = upper - this->mass.begin();
	int left_ind = right_ind - 1;
	double f = (mass - this->mass[left_ind]) / (this->mass[right_ind] - this->mass[left_ind]);
	return f * this->prob[right_ind] + (1-f) * this->prob[left_ind];
}

double MassPrior::logpdf(const double mass)
{
	return this->pdf(mass);
}

void MassPrior::rvs(std::vector<double>& samples)
{
	int n = samples.size();
	for (int i=0;i<n;i++)
	{
		double r = this->dist(this->gen);
		auto upper = std::upper_bound(this->cumsum.begin(), this->cumsum.end(), r);
		int right_ind = upper - this->cumsum.begin();
		int left_ind = right_ind - 1;
		double f = (r - this->cumsum[left_ind]) / (this->cumsum[right_ind] - this->cumsum[left_ind]);
		samples[i] = f * this->mass[right_ind] + (1-f) * this->mass[left_ind];
	}
}

void MassPrior::print()
{
	int n = this->mass.size();
	for (int i=0; i<n; i++)
	{
		std::cout << this->mass[i] << ","\
		<< this->prob[i] << ","\
		<< this->cumsum[i] << std::endl;
	}
}

TinkerPrior::TinkerPrior()
{
	double z, m, p;
	std::ifstream infile("massprior.txt");
	this->priormap = std::map<long, MassPrior>();
	std::vector<double> mass(410);
	std::vector<double> prob(410);

	infile >> z >> m >> p;
	double prevz = z;
	int i = 0;
	mass[i] = m;
	prob[i] = p;
	i += 1;
	while (infile >> z >> m >> p)
	{
		if (z != prevz)
		{
			this->priormap.emplace(TinkerPrior::snap(prevz), MassPrior(mass, prob));
			i = 0;
			mass[i] = m;
			prob[i] = p;
			i += 1;
			prevz = z;
		}
		else{
			mass[i] = m;
			prob[i] = p;
			i += 1;
			// std::cout >> z >> "," >> mass >> "," >> prob >> std::endl;			
		}
	}
	this->priormap.emplace(TinkerPrior::snap(z), MassPrior(mass, prob));
}

double TinkerPrior::pdf(double z, const double mass)
{
	return this->priormap.find(TinkerPrior::snap(z))->second.pdf(mass);
}

double TinkerPrior::logpdf(double z, const double mass)
{
	return this->priormap.find(TinkerPrior::snap(z))->second.logpdf(mass);
}

void TinkerPrior::rvs(double z, std::vector<double>& samples)
{
	auto search = this->priormap.find(TinkerPrior::snap(z));
	search->second.rvs(samples);
}

MassPrior TinkerPrior::fetch(double z)
{
	return this->priormap.find(TinkerPrior::snap(z))->second;
}

long TinkerPrior::snap(const double z)
{
	return (long) z * 100;
}
