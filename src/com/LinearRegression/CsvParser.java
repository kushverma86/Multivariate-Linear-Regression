package com.LinearRegression;


import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Scanner;

public class CsvParser {
    //ArrayList<Row> ob1=new ArrayList<Row>();

    private ArrayList<Instance> Instances;           //ArrayList of instance(points)
    private int attCount;

    public ArrayList<Instance> getdata() {

        Instances=new ArrayList<>();
        String path;
        Scanner sc=new Scanner(System.in);
        path=sc.nextLine();
        /* To read data from file*/
        BufferedReader br = null;
        Constant.line = "";

        try {
            int i = 0;
            br = new BufferedReader(new FileReader(path));

            while ((Constant.line = br.readLine()) != null) {
                //to skip the header line
                if (i == 0) {

                    i++;
                    continue;
                }


                String[] str = Constant.line.split(",");

                attCount=str.length;

               Instance r = new Instance(str);        //Single Point

                Instances.add(r);

                i++;

            }

            Iterator itr = Instances.iterator();
            while (itr.hasNext()) {
                System.out.println(itr.next());
            }



//            try {
//
//               // System.out.println(Instances.get(0).get(3));
//                // System.out.println(ob1.get(1).get(4));
//            } catch (IndexOutOfBoundsException e) {
//                System.out.println("Instance Not exist");
//            }

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        return Instances;

    }

    int getAttCount()
    {
        return attCount;
    }

}
