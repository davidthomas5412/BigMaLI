#ifndef IMPORTANCE_SAMPLING_H
#define IMPORTANCE_SAMPLING_H

#include <random>
#include "prior.hpp"

class ImportanceSampling
{
    public:
        static double p1(double lobs, double lum, double sigma);
        static double p2(double lum, double mass, double a1, double a2, double a3, double a4, double S, double z);
        static double p3(double mass, double z, TinkerPrior& prior);
        static double q1(double lum, double lobs, double sigma);
        static double q2(double mass, double lum, double a1, double a2, double a3, double a4, double S, double z);
        static double log_multi_importance_sampling(std::vector<double>& lum_obs, std::vector<double>& zs, double a1, double a2, double a3, double a4, \
            double S, TinkerPrior& prior, double nsamples);
        static double fast_lognormal(double mu, double sigma, double x);
        static constexpr double sigma = 0.05;
        static constexpr double lambda = 5.6578015811698101;
};

#endif