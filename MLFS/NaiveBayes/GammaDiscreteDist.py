# Translate from C#

# #region Imports

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Maths;

#endregion

namespace Statistics {

    internal class DiscreteDistrib
    {

        #region Fields

        private Combinatory comb;

        #endregion

        #region Constructors

        internal DiscreteDistrib()
        {
            comb = new Combinatory();
        }

        #endregion

        #region internal Methods

        #region Probability

        internal double Binomial(int x, int n, double p, bool acum)
        {
            if (acum)
            {
                double sum = 0.0;
                for (int i = 0; i <= x; i++) { sum += Binomial(i, n, p); }
                return sum;
            }
            else
            {
                return Binomial(x, n, p);
            }
        }

        internal double Hypergeometric(int x, int N, int n, int d, bool acum)
        {
            if (acum)
            {
                double sum = 0.0;
                for (int i = 0; i <= x; i++) { sum += Hypergeometric(i, N, n, d); }
                return sum;
            }
            else
            {
                return Hypergeometric(x, N, n, d);
            }
        }

        internal double Poisson(int x, double lambda, bool acum)
        {
            if (acum)
            {
                double sum = 0.0;
                for (int i = 0; i <= x; i++) { sum += Poisson(i, lambda); }
                return sum;
            }
            else
            {
                return Poisson(x, lambda);
            }
        }

        internal double HurdlePoisson(int x, double alpha, double lambda, bool acum)
        {
            if (x > 160) { x = 160; } //max value
            if (acum)
            {
                double sum = 0.0;
                for (int i = 0; i <= x; i++) { sum += HurdlePoisson(i, alpha, lambda); }
                return sum;
            }
            else
            {
                return HurdlePoisson(x, alpha, lambda);
            }
        }



        internal double NegativeBinomial(int x, int n, double p, bool acum)
        {
            if (acum)
            {
                double sum = 0.0;
                for (int i = 0; i <= x; i++) { sum += NegativeBinomial(i, n, p); }
                return sum;
            }
            else
            {
                return NegativeBinomial(x, n, p);
            }
        }

        #endregion

        #region Quantil

        internal int InverseBinomial(double prob, int n, double p)
        {
            double acum = 0.0;
            for (int i = 0; i <= n; i++)
            {
                acum += Binomial(i, n, p);
                if (acum >= prob) { return i; }
            }
            return 0;
        }


        #endregion

        #region Statistics

        #region Mean

        internal double GetBinomialMean(int n, int p)
        {
            return n * p;
        }

        internal double GetHypergeometricMean(int N, int n, int d)
        {
            return (n * d) / N;
        }

        internal double GetPoissonMean(double lambda)
        {
            return lambda;
        }

        internal double GetNegativeBinomialMean(int x, int n, double p)
        {
            return x / p;
        }

        #endregion

        #region Standard Deviation

        internal double GetBinomialStDev(int n, int p)
        {
            return Math.Sqrt(n * p * (1 - p));
        }

        internal double GetHypergeometricStDev(int N, int n, int d)
        {
            return (n * (d / N) * (1 - (d / N)) * (N - n)) / (N - 1);
        }

        internal double GetPoissonStDev(double lambda)
        {
            return Math.Sqrt(lambda);
        }

        internal double GetNegativeBinomialStDev(int x, int n, double p)
        {
            return (x * (1 - p)) / Math.Pow(p, 2);
        }

        #endregion

        #endregion

        #endregion

        #region Private Methods

        #region Distributions

        private double Binomial(int x, int n, double p)
        {
            return comb.Combinations(n, x) * Math.Pow(p, x) * Math.Pow((1 - p), (n - x));
        }


        private double Poisson(int x, double lambda)
        {
            return (Math.Exp(-lambda) * Math.Pow(lambda, x)) / comb.Factorial(x);
        }

        private double HurdlePoisson(int x, double alpha, double lambda)
        {
            if (x == 0) { return alpha; }
            return ((1 - alpha) * Math.Pow(lambda, x)) / (comb.Factorial(x) * (Math.Exp(lambda) - 1));
        }

        private double Hypergeometric(int x, int N, int n, int d)
        {
            return (comb.Combinations(d, x) * comb.Combinations(N - d, N - x)) / comb.Combinations(N, n);
        }

        private double NegativeBinomial(int x, int n, double p)
        {
            return comb.Combinations(x - 1, n - 1) * Math.Pow(p, n) * Math.Pow(1 - p, x - n);
        }

        #endregion

        # region Log Likelihood

        //n : number of periods - n0 : number of periods without event - t : total of events
        private double NegativeLogLikelihood(double alpha, double lambda, int n, int n0, int t)
        {
            return -n0 * Math.Log(alpha) - (n - n0) * Math.Log((1 - alpha) / (Math.Exp(lambda) - 1)) - t * Math.Log(lambda);
        }

        internal void MinimizeLikelihood(double alpha, ref double lambda, int maxLambda, int n, int n0, int t)
        {
            double minNll = double.MaxValue;
            double nll = 0;

            for (int l = 1; l < maxLambda; l++)
            {
                nll = NegativeLogLikelihood(alpha, (double)l, n, n0, t);
                if (nll < minNll) { lambda = l; minNll = nll; }
            }

            double lam = lambda - 1;
            double Lambda = lambda;
            while (lam <= lambda + 1)
            {
                nll = NegativeLogLikelihood(alpha, lam, n, n0, t);
                if (nll < minNll) { Lambda = lam; minNll = nll; }
                lam = lam + 0.1;
            }
            lambda = Lambda;

            #region Obsolete

            /*
               Random rand = new Random();
               double a, l;

               //explotarion: montecarlo
               for (int i = 0; i < 100; i++) {
                   a = rand.Next(99) / 100.0;
                   l = rand.Next(maxLambda);
                   TestParameters(a, l, ref alpha, ref lambda, n, n0, t, ref minNll);
               }

               //explotation: gradient
               for (int i = 0; i < 100; i++) {
                   if (alpha > 0.01) { TestParameters(alpha - 0.01, lambda, ref alpha, ref lambda, n, n0, t, ref minNll); }
                   if (alpha < 0.99) { TestParameters(alpha + 0.01, lambda, ref alpha, ref lambda, n, n0, t, ref minNll); }
                   if (lambda > 1) { TestParameters(alpha, lambda - 1, ref alpha, ref lambda, n, n0, t, ref minNll); }
                   if (lambda < maxLambda) { TestParameters(alpha, lambda + 1, ref alpha, ref lambda, n, n0, t, ref minNll); }
               }
               */
            #endregion

        }

        private void TestParameters(double a, double l, ref double alpha, ref double lambda, int n, int n0, int t, ref double minNll)
        {
            double nll = NegativeLogLikelihood(a, l, n, n0, t);
            if (nll < minNll) { alpha = (double)a / 100.0; lambda = l; minNll = nll; }
        }

        #endregion

        #endregion

    }
}

