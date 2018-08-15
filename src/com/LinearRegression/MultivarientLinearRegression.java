package com.LinearRegression;


import java.util.Arrays;

public class MultivarientLinearRegression {
    private double[][] featureMatrix;
    double[] outputVector;
    int _nrows;
    int _ndims;
    double[] weights;
    double[] transposeWeights;
    double cost;
    int rounds;


    MultivarientLinearRegression(double[][] featureMatrix,double[] outputVector,int _nrows,int _ndims)
    {
      this.featureMatrix=featureMatrix;
      this.outputVector=outputVector;
      this._nrows=_nrows;
      this._ndims=_ndims;
      this.weights=new double[_ndims];
      this.transposeWeights=new double[_ndims];
    }

    void init(int MAX_ITER){

        this.rounds=MAX_ITER;
        for(int i=0;i<_ndims;i++)
        {
           // int c=(int)(10*Math.random());

            weights[i]=0;
        }

        transposeWeights=getTranspose(weights);

        cost=getCost();

        System.out.println();
        System.out.println("Final Loss:"+cost);

    }

//    double[] getRow(int k)
//    {
//        double[] x={0};
//        for(int j=0;j<_ndims;j++)
//        {
//            x[j]=featureMatrix[k][j];
//        }
//
//        return x;
//    }

    double updatecost()
    {
        double squared_error=0;
        double diff;

        for(int i=0;i<_nrows;i++)
        {
            double[] x=featureMatrix[i];
            diff=Hypothesis_result(transposeWeights,x)-outputVector[i];

            squared_error+=diff*diff;
        }

        double k=squared_error/(2*_nrows);

      return k;
    }

    double getCost()
    {
       double squared_error=0;
       double diff;

       for(int i=0;i<_nrows;i++)
       {
           double[] x=featureMatrix[i];
           diff=Hypothesis_result(transposeWeights,x)-outputVector[i];

           squared_error+=diff*diff;
       }

       cost=squared_error/(2*_nrows);

        System.out.println("cost:"+cost);

       if(cost==0)
           return cost;

       PerformGradientDescent();

        return updatecost();


    }


    double diff(int j)
    {
        double diff;
        double sum=0;
        double k;
        for(int i=0;i<_nrows;i++)
        {
           diff=Hypothesis_result(transposeWeights,featureMatrix[i])-outputVector[i];
           k=featureMatrix[i][j];
           sum+=diff*k;
        }
        return sum;
    }


    void PerformGradientDescent()
    {
        double[] temp=new double[_ndims];
        double learningRate=0.01;
        int round=0;
        while(true){

            round++;
            for(int i=0;i<_ndims;i++)
            {
                double diffError=diff(i);

                temp[i]=weights[i]-((learningRate/_nrows)*diffError);
            }

            if(Arrays.equals(temp,weights)||round==rounds)
            {
                System.out.println(round);
                break;
            }



            updateWeights(temp);


            double squared_error=0;
            double diff;

            for(int i=0;i<_nrows;i++)
            {
                double[] x=featureMatrix[i];
                diff=Hypothesis_result(transposeWeights,x)-outputVector[i];

                squared_error+=diff*diff;
            }

            cost=squared_error/(2*_nrows);

            System.out.println("cost:"+cost);


            //getCost();

//            System.out.println();
//            for(int i=0;i<_ndims;i++)
//                System.out.print(" "+weights[i]);


        }

//        for(int i=0;i<_ndims;i++)
//            System.out.print(" "+weights[i]);



        double[] wt=getFinalWeights();


    }



    void updateWeights(double[] temp)
    {
        for(int i=0;i<_ndims;i++)
        {
            weights[i]=temp[i];
        }

        transposeWeights= getTranspose(weights);

    }

    double Hypothesis_result(double[] transposeWeights,double[] X)
    {
        double result=0;
        for(int i=0;i<_ndims;i++)
        {
            double product=transposeWeights[i]*X[i];
            result+=product;
        }
        return result;
    }

    double[] getTranspose(double weights[])
    {
     return weights;
    }

    public double[] getFinalWeights() {


        return weights;
    }
}
