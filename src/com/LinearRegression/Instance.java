package com.LinearRegression;

import java.util.ArrayList;

public class Instance extends ArrayList<Double> {

      ArrayList<Double> ob;
   // static ArrayList<ArrayList<Double>> ob1=new ArrayList<ArrayList<Double>>();


    Instance(String[] str){

        ob=new ArrayList<>();

        for(int k=0;k<str.length;k++)
        {

           ob.add(Double.parseDouble(str[k]));
        }
       // ob1.add(ob);



    }

    public Double get(int index)
    {
        return ob.get(index);
    }

    /*Object conversion into string*/

    @Override
    public String toString() {
        String output = "[";
        for(double val : ob){
            output +=" " + val ;
        }
        output = output + " ]";
        return output;
    }

  /* public String toString(){

       return ob.toString();


   } */



}
