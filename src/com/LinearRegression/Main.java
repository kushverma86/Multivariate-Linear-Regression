package com.LinearRegression;

import java.util.ArrayList;

public class Main {
    public static void main(String[] args) {
        System.out.println("Enter the file name to load data :");
        CsvParser csvp=new CsvParser();               // parsing Csv File
        ArrayList<Instance> Instances= csvp.getdata();                               // get the instances arraylist
        int _ndims=csvp.getAttCount();
        System.out.println(_ndims);
        int _nrows=Instances.size();
        System.out.println(_nrows);
        InputOutputVectors IOV=new InputOutputVectors(Instances,_ndims);
        double[][] featureMatrix=IOV.getX_inputs();
        double[] outputVecor=IOV.getY_outputs();
        MultivarientLinearRegression mlr=new MultivarientLinearRegression(featureMatrix,outputVecor,_nrows,_ndims);
        mlr.init(100);
    }
}
