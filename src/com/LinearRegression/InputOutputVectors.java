package com.LinearRegression;

import java.lang.reflect.Array;
import java.util.ArrayList;

public class InputOutputVectors {

    ArrayList<Instance> Instances;
    int _ndims;
    int _nrows;
    double[][] x_inputs;
    double[] y_outputs;

  InputOutputVectors(ArrayList<Instance> Instances,int _ndims)
  {
      this.Instances=Instances;
      this._ndims=_ndims;
      this._nrows=Instances.size();
      this.x_inputs=new double[_nrows][_ndims];
      this.y_outputs=new double[_nrows];
  }

  public double[][] getX_inputs()
  {
      for(int i=0;i<_nrows;i++)
      {
          for(int j=0;j<_ndims;j++)
          {
              if(j==0)
                  x_inputs[i][j]=1;
             else
                x_inputs[i][j]=Instances.get(i).get(j-1);


          }
      }



      return x_inputs;
  }

    public double[] getY_outputs() {

      for(int i=0;i<_nrows;i++)
      {
          y_outputs[i]=Instances.get(i).get(_ndims-1);
      }
        return y_outputs;
    }
}
