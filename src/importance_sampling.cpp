#define _USE_MATH_DEFINES

#include "importance_sampling.hpp"
#include "prior.hpp"
#include <cmath>
#include <iostream>

double ImportanceSampling::fast_lognormal(double mu, double sigma, double x)
{
    return (1/(x * sigma * sqrt(2 * M_PI))) * exp(- 0.5 * pow(log(x) - log(mu), 2) / pow(sigma, 2));
}

double ImportanceSampling::p1(double lobs, double lum, double sigma)
{
    return ImportanceSampling::fast_lognormal(lum, sigma, lobs);
}

double ImportanceSampling::p2(double lum, double mass, double a1, double a2, double a3, double a4, double S, double z)
{
    double mu_lum = exp(a1) * pow((mass / a3), a2) * pow(1+z, a4);
    return ImportanceSampling::fast_lognormal(mu_lum, S, lum);
}

double ImportanceSampling::p3(double mass, double z, TinkerPrior& prior)
{
    return prior.logpdf(z, mass);
}

double ImportanceSampling::q1(double lum, double lobs, double sigma)
{
    return ImportanceSampling::fast_lognormal(lobs, sigma, lum);
}


double ImportanceSampling::q2(double mass, double lum, double a1, double a2, double a3, double a4, double S, double z)
{
    double mu_mass = a3 * pow(lum / (exp(a1) * pow(1 + z, a4)), 1 / a2);
    return ImportanceSampling::fast_lognormal(mu_mass, S, mass);
}

double ImportanceSampling::log_multi_importance_sampling(std::vector<double>& lum_obs, std::vector<double>& zs, double a1, double a2, double a3, double a4, \
            double S, TinkerPrior& prior, double nsamples)
{
    std::default_random_engine generator; //TODO: put somewhere else
    std::lognormal_distribution<double> lumdist, massdist;
    int n = lum_obs.size();
    double rev_S = ImportanceSampling::lambda * S;
    double result = 0;
    double lum, mu_mass, mass;
    for (int i=0; i<n; i++)
    {

        double tmp = 0;
        //make dist
        lumdist = std::lognormal_distribution<double>(log(lum_obs[i]), sigma);

        for (int j=0; j<nsamples; j++)
        {
            // indexed by j
            lum = lumdist(generator);
            mu_mass = a3 * pow(lum / (exp(a1) * pow(1 + zs[i], a4)), 1 / a2);

            massdist = std::lognormal_distribution<double>(log(mu_mass), rev_S);
            mass = massdist(generator);
            tmp += (ImportanceSampling::p1(lum_obs[i], lum, sigma) * 
                    ImportanceSampling::p2(lum, mass, a1, a2, a3, a4, S, zs[i]) *
                    ImportanceSampling::p3(mass, zs[i], prior)) /
                    (ImportanceSampling::q1(lum, lum_obs[i], sigma) * 
                     ImportanceSampling::q2(mass, lum, a1, a2, a3, a4, rev_S, zs[i])
                     );
        }
        result += log(tmp / nsamples);
    }
    return result;
}
