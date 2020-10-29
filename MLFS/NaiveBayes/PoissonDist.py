#region Imports

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Maths;

#endregion

namespace Statistics {

    /** Method:  Class for any Poisson distribution calculations */
    internal class PoissonCalc {

        #region Fields

        private List<double> data;
        private double loadFactor;

        private List<double>[] dataSplit;
        private double[] Lambda;
        private double[] Alpha;
        private double alpha;
        private double lambda;

        private DiscreteDistrib dd;
        private StatFunctions stat;
        private RootSearch rs;
        private RndGenerator rg;
        private Functions func;

        private double max;
        private double epsilon;
        private int maxIt;

        private MethodType method;

        private double[] weights;
        private int maxWeighting;
        private int minWeighting;
        private double minWeight;
        private double firstNonZero;


        #endregion

        #region Constructor

        /** Method:  Constructor */
        internal PoissonCalc(double epsilon, int maxIt, int forgetInitPeriods, int forgetEndPeriods, double forgetEndProportion)
        {
            this.dd = new DiscreteDistrib();
            this.stat = new StatFunctions();
            this.epsilon = epsilon;
            this.maxIt = maxIt;
            this.rs = new RootSearch(epsilon, -1, maxIt);
            this.rg = new RndGenerator();
            this.func = new Functions();
            this.method = MethodType.Poisson;
            this.maxWeighting = forgetInitPeriods;
            this.minWeighting = forgetEndPeriods;
            this.minWeight = forgetEndProportion;
            this.Lambda = new double[3];
            this.dataSplit = new List<double>[3];
            this.weights = new double[3];
            this.firstNonZero = 0;

        }

        #endregion

        #region Properties

        internal List<double> Data
        {
            get { return data; }
        }

        internal double GetAlpha(int i)
        {
            return Alpha[i];
        }

        internal double GetLambda(int i)
        {
            return Lambda[i];
        }

        internal MethodType Method
        {
            get { return method; }
            set { method = value; }
        }

        internal int MaxWeighting
        {
            get { return maxWeighting; }
            set { maxWeighting = value; }
        }

        internal int MinWeighting
        {
            get { return minWeighting; }
            set { minWeighting = value; }
        }

        internal double MinWeight
        {
            get { return minWeight; }
            set { minWeight = value; }
        }

        internal double FirstNonZero
        {
            get { return firstNonZero; }
            set { firstNonZero = value; }
        }

        #endregion

        #region internal Methods

        /** Method:  Load data for calculation */
        internal void LoadData(List<double> data)
        {
            this.data = data;
            bool useMemory = (maxWeighting > 0 && minWeighting > 0 && minWeighting < data.Count && minWeight > 0 && minWeight < 1);
            CalculateParameters(data, useMemory);
            CalculateMax();
        }

        /** Method:  Probability of a number of events (or quantity) to happen in span period */
        internal double Probability(double x, bool acum)
        {
            double prob = 0;
            double p = 0;
            //for (int i = 0; i < 3; i++) {
            switch (method)
            {
                //case MethodType.Poisson: p = dd.Poisson((int)x, Lambda[i], acum); break;
                case MethodType.Hurdle: p = dd.HurdlePoisson((int)x, alpha, lambda, acum); break;
            }
            //prob += weights[i] * p;
            //}
            prob = p;
            if (double.IsInfinity(p)) { prob = 0; }
            return prob;
        }

        private double Probability(double x)
        {
            if (x >= this.max) { return 1.00; }
            return Probability(x, true);
        }

        /** Method:  Quantile for a certain probability (inverse function) */
        internal double Quantile(double p)
        {
            if (p >= 0.99) { return this.max; }
            int it = 0;
            double q1 = rs.MonotoneBisection(Probability, true, 0, max, p, epsilon, ref it, maxIt);
            double q2 = q1 + 1;
            if (Math.Abs(Probability(q1) - p) <= Math.Abs(Probability(q2) - p)) { return Math.Round(q1); }
            else { return Math.Round(q2); }
        }

        /** Method:  Get the histogram of the distribution */
        /// <returns> the histogram </returns>
        internal Histogram GetHistogram()
        {
            Histogram hist = new Histogram(100);
            List<double> freqs = new List<double>();
            double p;
            double pTot = 0;
            for (int x = 0; x <= this.max; x++)
            {
                p = Probability(x, false);
                if (p <= 0.0005) { p = 0; }
                pTot += p;
                if (p > 0 && x > 0.5 && firstNonZero == 0) { firstNonZero = Math.Round(pTot, 3); }
                if (x == max && pTot < 1.00) { p += 1.0 - pTot; }
                freqs.Add(p);
            }
            hist.LoadDist(freqs);
            return hist;
        }

        #endregion

        #region Private Methods

        private void CalculateN0t(List<double> data, ref int n0, ref int t)
        {
            n0 = 0;
            t = 0;
            foreach (double d in data)
            {
                if (d <= 0) { n0++; } else { t = t + (int)d; }
            }
        }

        private void CalculateParameters(List<double> data, bool useMemory)
        {
            if (useMemory) { CalculateParametersUseMemory(data, maxWeighting, minWeighting, minWeight); }
            else { CalculateParameters(data); }
        }

        private void CalculateParameters(List<double> data)
        {
            int n = data.Count;
            int n0 = 0;
            int t = 0;
            CalculateN0t(data, ref n0, ref t);

            int maxLambda = 300;
            alpha = (double)n0 / (double)n;
            lambda = -1;
            dd.MinimizeLikelihood(alpha, ref lambda, maxLambda, n, n0, t);
        }

        private void CalculateParametersUseMemory(List<double> data, int forgetInitPeriods, int forgetEndPeriods, double forgetEndProportion)
        {
            double[] weights = func.CalcTempWeights(data.Count, forgetInitPeriods, forgetEndPeriods, forgetEndProportion);

            double n = 0;
            double n0 = 0;
            double t = 0;
            for (int i = 0; i < data.Count; i++)
            {
                n += weights[i];
                if (data[i] <= 0) { n0 += weights[i]; }
                else { t = t + data[i] * weights[i]; }
            }

            int maxLambda = 300;
            alpha = n0 / n;
            lambda = -1;
            int nInt = (int)Math.Round(n);
            int n0Int = (int)Math.Round(n0);
            int tInt = (int)Math.Round(t);
            dd.MinimizeLikelihood(alpha, ref lambda, maxLambda, nInt, n0Int, tInt);
        }

        private double CalculateMax()
        {
            this.max = func.Max(data);
            while (Probability(max) < 0.99)
            {
                if (max < 20) { max = max + 1; }
                else { max = max * 1.1; }
            }
            return max;
        }

        #endregion

        #region Obsolete

        private void SplitData(IList<double> rawData, int forgetInitPeriods, int forgetEndPeriods, double forgetEndProportion)
        {
            if (forgetInitPeriods == 0 && forgetEndPeriods == 0)
            {
                forgetInitPeriods = (int)(rawData.Count * 2.0 / 3.0);
                forgetEndPeriods = (int)(rawData.Count / 3.0);
                forgetEndProportion = 0.2;
            }
            dataSplit[0] = new List<double>();
            dataSplit[1] = new List<double>();
            dataSplit[2] = new List<double>();
            for (int i = rawData.Count - 1; i >= 0; i--)
            {
                if (i > rawData.Count - forgetInitPeriods) { dataSplit[0].Add(rawData[i]); } else if (i > rawData.Count - forgetEndPeriods) { dataSplit[1].Add(rawData[i]); } else { dataSplit[2].Add(rawData[i]); }
            }

            weights[0] = 1 * dataSplit[0].Count;
            weights[1] = ((1 - forgetEndProportion) / 2.0) * dataSplit[1].Count;
            weights[2] = forgetEndProportion * dataSplit[2].Count;
            weights = Normalize(weights);

        }

        /*
        private void CalculateLambdas(int span) {
            alpha = new double[3];
            switch (method) {
                case MethodType.Poisson:
                    for (int i = 0; i < 3; i++) {
                        lambda[i] = CalculateLambda(dataSplit[i], span);
                    }
                    break;
                case MethodType.Hurdle:
                    for (int i = 0; i < 3; i++) {
                        double alph = 0;
                        double lambd = 0;
                        CalculateAlphaLambda(dataSplit[i], span, ref alph, ref lambd);
                        alpha[i] = alph;
                        lambda[i] = lambd;
                    }
                    break;
            }

        }
        */

        private List<double> Group(IList<double> rawData, int group)
        {
            max = 0;
            data = new List<double>();
            int day = 0;
            double totPeriod = 0.0;
            double val;
            int nonZero = 0;
            for (int i = rawData.Count - 1; i >= 0; i--)
            {
                val = rawData[i];
                totPeriod += val;
                day++;
                if (day == group)
                {
                    data.Insert(0, totPeriod);
                    day = 0;
                    if (totPeriod > 0) { nonZero++; }
                    totPeriod = 0.0;
                }
            }
            if (day > 0)
            {
                data.Insert(0, totPeriod);
                if (totPeriod > 0) { nonZero++; }
                if (totPeriod > max) { max = totPeriod; }
            }
            loadFactor = (double)nonZero / (double)data.Count;
            return data;
        }

        private double CalculateLambda(IList<double> data)
        {
            double lambda = stat.Mean(data);
            return lambda;
        }

        private double[] Normalize(IList<double> W)
        {
            double[] P = new double[W.Count];
            double sumW = 0;
            foreach (double w in W) { sumW += w; }
            for (int i = 0; i < W.Count; i++) { P[i] = W[i] / sumW; }
            return P;
        }

        private double CalculateLambda(IList<double> rawData, int span)
        {
            double sumLambda = 0;
            double lambda, maX;
            List<double> rawSampData;
            List<double> grSamp;
            for (int i = 0; i < maxIt; i++)
            {
                rawSampData = Resample(rawData);
                grSamp = Group(rawSampData, span);
                lambda = stat.Mean(grSamp);
                maX = func.Max(grSamp);
                if (maX > this.max) { this.max = maX; }
                sumLambda += lambda;
            }
            return sumLambda / (double)maxIt;
        }

        /*
        private void CalculateAlphaLambda(IList<double> rawData, int span, ref double alpha, ref double lambda) {
            this.max = func.Max(rawData);

            int n0 = 0;
            int t = 0;
            foreach (double d in grData) {
                if (d == 0) { n0++; }
                if (d > 0) { t = t + (int)d; }
            }
            int maxLambda = 300;
            int n = rawData.Count;
            dd.MinimizeLikelihood(ref alpha, ref lambda, maxLambda, n, n0, t);
        }
        */
        /*
        private double CalculateMax() {
            while (Probability(max) < 0.99) {
                if (max < 20) { max = max + 1; } else { max = max * 1.1; }
            }
            return max;
        }
        */
        private List<double> Resample(IList<double> rawData, IList<double> W)
        {
            List<double> resamp = new List<double>();
            int n = rawData.Count;
            int index;
            double u;

            while (resamp.Count < n)
            {
                index = rg.NextInt(0, n - 1);
                u = rg.NextDouble(0, 1);
                if (u <= W[index]) { resamp.Add(rawData[index]); }
            }
            return resamp;
        }

        private List<double> Resample(IList<double> rawData)
        {
            List<double> resamp = new List<double>();
            int index;
            while (resamp.Count < rawData.Count)
            {
                index = rg.NextInt(0, rawData.Count - 1);
                resamp.Add(rawData[index]);
            }
            return resamp;
        }

        #endregion

        #region Enums

        internal enum MethodType { Poisson, Hurdle };

        #endregion
    }
}

